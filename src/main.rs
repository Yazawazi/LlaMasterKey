use std::{collections::HashMap, str::FromStr, sync::Arc};
use std::string::ToString;
use anyhow::Context;
use axum::{
  body::Body,
  extract::{Path, Query, State},
  http::{
    uri::{PathAndQuery, Scheme},
    HeaderValue,
  },
  response::{IntoResponse, Response},
  routing, Json, Router,
};
use config::*;
use hyper::{header::CONTENT_TYPE, Request, StatusCode, Uri};
use itertools::Itertools;
use sha2::{Sha256, Digest};
use state::{HyperClient, Server, ServerState};
use tokio::net::TcpListener;
use tower_http::{compression::CompressionLayer, limit::RequestBodyLimitLayer};
use once_cell::sync::Lazy;

mod config;
mod state;

pub const VERSION: &str = env!("CARGO_PKG_VERSION");

static LMK_KEY: Lazy<String> = Lazy::<String>::new(|| {
  if std::env::var("LLAMA_PASS_TOKEN").is_err() {
    std::env::set_var("LLAMA_PASS_TOKEN", "Jokes like trying to redo it don't work.");
  }
  std::env::var("LLAMA_PASS_TOKEN").unwrap()
});

const WARN_MESSAGE: &str = "THIS PERMANENT KEY ONLY ALLOWS THE INSTANCE TO SEND REQUESTS TO THE KUMO LMK SERVER, ANY REQUESTS FROM OUTSIDE MAY RESULT IN MORE BILLING.";

fn create_token(user_id: i64) -> String {
  let sign_hasher = Sha256::new()
      .chain_update("key")
      .chain_update(user_id.to_string())
      .chain_update(LMK_KEY.as_str())
      .finalize();
  let sign = base16ct::lower::encode_string(&sign_hasher);
  format!("{user_id},{WARN_MESSAGE},{sign}")
}

fn verify_token(token: &str) -> Option<i64> {
  let (user_id, warn_message, sign): (&str, &str, &str) = token.split(',').collect_tuple::<(_, _, _)>()?;
  if warn_message != WARN_MESSAGE {
    return None;
  }
  let user_id = user_id.parse::<i64>().ok()?;
  let sign_hasher = Sha256::new()
      .chain_update("key")
      .chain_update(user_id.to_string())
      .chain_update(LMK_KEY.as_str())
      .finalize();
  let expected_sign = base16ct::lower::encode_string(&sign_hasher);
  if sign != expected_sign {
    return None;
  }
  Some(user_id)
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
  // ignore errors when loading dotenv
  let _ = dotenvy::dotenv();

  if std::env::var("LMK_LOG").is_err() {
    std::env::set_var("LMK_LOG", "info");
  }
  pretty_env_logger::init_custom_env("LMK_LOG");

  log::info!("llamakey v{}", VERSION);

  let state = Arc::new(Server::new().context("Failed to init server state")?);

  let service = Router::new()
    .route("/lmk/env", routing::get(lmk_env))
    .route("/vectara/*path", routing::any(vectara))
    .route("/:llm/*path", routing::any(catch_all))
    .layer(CompressionLayer::new())
    .layer(RequestBodyLimitLayer::new(512 * 1024 * 1024))
    .with_state(state.clone())
    .into_make_service();

  log::info!("Serve server at: {}", state.config.address);
  axum::serve(
    TcpListener::bind(&state.config.address)
      .await
      .with_context(|| format!("Failed to serve server at {}", &state.config.address))?,
    service,
  )
  .await
  .context("Failed to start server")?;

  Ok(())
}

#[derive(Default)]
pub enum EnvFormat {
  Json,
  #[default]
  Dotenv,
}

impl EnvFormat {
  pub fn parse(value: &str) -> EnvFormat {
    match value {
      "json" => Self::Json,
      "dotenv" => Self::Dotenv,
      _ => Self::Json,
    }
  }
}

pub async fn lmk_env(
  State(state): ServerState,
  Query(query): Query<HashMap<String, String>>,
  request: Request<Body>,
) -> Response {
  let Some(auth) = request.headers().get("authorization") else {
    return (StatusCode::BAD_REQUEST, "No `Authorization` header").into_response();
  };
  let Some(requested_time) = request.headers().get("x-requested-at") else {
    return (StatusCode::BAD_REQUEST, "No `x-requested-at` header").into_response();
  };
  let Some(user_id) = request.headers().get("x-user-id") else {
    return (StatusCode::BAD_REQUEST, "No `x-user-id` header").into_response();
  };
  let current_time = chrono::Utc::now().timestamp();
  let requested_time = requested_time.to_str().unwrap().parse::<i64>().unwrap();
  if (current_time - requested_time).abs() > 60 {
    return (StatusCode::BAD_REQUEST, "Invalid `x-requested-at` header").into_response();
  }
  let user_id = user_id.to_str().unwrap();
  if user_id.parse::<i64>().is_err() {
    return (StatusCode::BAD_REQUEST, "Invalid `x-user-id` header").into_response();
  }
  let hasher = Sha256::new()
    .chain_update("generation")
    .chain_update(user_id)
    .chain_update(requested_time.to_string())
    .chain_update(LMK_KEY.as_str())
    .finalize();

  if auth.to_str().unwrap() != base16ct::lower::encode_string(&hasher) {
    return (StatusCode::BAD_REQUEST, "Invalid `Authorization` header").into_response();
  }
  
  let auth_key = create_token(user_id.parse::<i64>().unwrap());
  
  let format = query
    .get("format")
    .map(|v| EnvFormat::parse(v))
    .unwrap_or_default();
  match format {
    EnvFormat::Json => {
      let envs: HashMap<_, _> = state.config.user_env(Some(auth_key)).into_iter().collect();
      (StatusCode::OK, Json(envs)).into_response()
    },
    EnvFormat::Dotenv => (StatusCode::OK, state.config.user_env_file(Some(auth_key))).into_response(),
  }
}

fn rewrite_query(query: &mut Vec<(String, String)>, key: &str, new_value: &str) {
  if let Some(pos) = query.iter().position(|(k, _v)| k == key) {
    query.remove(pos);
    query.push((key.to_string(), new_value.to_string()));
  }
}

async fn body_query(request: Request<Body>) -> (Request<Body>, Option<Vec<(String, String)>>) {
  let (parts, body) = request.into_parts();
  let bytes = axum::body::to_bytes(body, usize::MAX).await.unwrap();
  let body = Body::from(bytes.clone());
  (
    Request::from_parts(parts, body),
    serde_urlencoded::from_bytes(&bytes).ok(),
  )
}

pub async fn vectara(
  State(state): ServerState,
  Path(path): Path<String>,
  mut request: Request<Body>,
) -> Response {
  let Some(auth) = request.headers().get("authorization") else {
    return (StatusCode::BAD_REQUEST, "No `Authorization` header").into_response();
  };
  
  let token = auth.to_str().unwrap();
  let Some(_user_id) = verify_token(token) else {
    return (StatusCode::BAD_REQUEST, "Invalid `Authorization` header").into_response();
  };
  
  
  let Some(token) = &state.config.vectara_token else {
    return (
      StatusCode::INTERNAL_SERVER_ERROR,
      "Server has no configured token for vectara",
    )
      .into_response();
  };

  // base URL
  let url = if path == "oauth2/token" {
    Uri::from_str(&format!(
      "vectara-prod-{}.auth.us-west-2.amazoncognito.com",
      token.customer_id
    ))
    .expect("Building vectara target url")
  } else {
    Uri::from_static(ModelEndpoint::Vectara.base_url())
  };

  let (query, is_body) = if request
    .headers()
    .get(CONTENT_TYPE)
    .and_then(|v| v.to_str().ok())
    .is_some_and(|value| value.starts_with("application/x-www-form-urlencoded"))
  {
    let (new_request, query) = body_query(request).await;
    request = new_request;
    (query, true)
  } else {
    let query = request
      .uri()
      .query()
      .and_then(|query| serde_urlencoded::from_str::<Vec<(String, String)>>(query).ok());
    (query, false)
  };

  let uri = request.uri();
  let mut uri_parts = uri.clone().into_parts();

  // rewrite query
  if let Some(mut query) = query {
    rewrite_query(&mut query, "c", &token.customer_id);

    if path == "oauth2/token" {
      rewrite_query(&mut query, "client_id", &token.client_id);
      rewrite_query(&mut query, "client_secret", &token.client_secret);
    }

    let query = serde_urlencoded::to_string(query).ok().unwrap_or_default();

    if is_body {
      *request.body_mut() = Body::from(query);
    } else {
      let path = uri.path();
      uri_parts.path_and_query = Some(PathAndQuery::from_str(&format!("{path}?{query}")).unwrap());
    }
  }
  *request.uri_mut() = Uri::from_parts(uri_parts).unwrap();

  // rewrite header
  request.headers_mut().insert(
    "customer-id",
    HeaderValue::from_str(&token.customer_id).unwrap(),
  );

  reverse_proxy(&state.hyper_client, request, url, None, Some("/vectara")).await
}

pub async fn catch_all(
  State(state): ServerState,
  Path((llm, _path)): Path<(String, String)>,
  request: Request<Body>,
) -> Response {
  let Some(auth) = request.headers().get("authorization") else {
    return (StatusCode::BAD_REQUEST, "No `Authorization` header").into_response();
  };

  let token = auth.to_str().unwrap();
  let Some(_user_id) = verify_token(token) else {
    return (StatusCode::BAD_REQUEST, "Invalid `Authorization` header").into_response();
  };

  let (url, key) = match llm.as_str() {
    "openai" => (
      Uri::from_static(ModelEndpoint::OpenAI.base_url()),
      &state.config.openai_api_key,
    ),
    "anyscale" => (
      Uri::from_static(ModelEndpoint::Anyscale.base_url()),
      &state.config.anyscale_api_key,
    ),
    "cohere" => (
      Uri::from_static(ModelEndpoint::Cohere.base_url()),
      &state.config.cohere_api_key,
    ),
    "huggingface" => (
      Uri::from_static(ModelEndpoint::HuggingFace.base_url()),
      &state.config.huggingface_api_key,
    ),
    "perplexity" => (
      Uri::from_static(ModelEndpoint::Perplexity.base_url()),
      &state.config.perplexity_api_key,
    ),
    _ => return (StatusCode::BAD_REQUEST, "Invalid token").into_response(),
  };

  let Some(key) = key else {
    return (
      StatusCode::INTERNAL_SERVER_ERROR,
      "Server has no configured key for this endpoint",
    )
      .into_response();
  };

  reverse_proxy(
    &state.hyper_client,
    request,
    url,
    Some(key),
    Some(&format!("/{llm}")),
  )
  .await
}

async fn reverse_proxy(
  client: &HyperClient,
  mut request: Request<Body>,
  uri: Uri,
  key: Option<&str>,
  remove_path: Option<&str>,
) -> hyper::Response<Body> {
  let mut uri_parts = request.uri().clone().into_parts();
  if uri_parts.scheme.is_none() {
    uri_parts.scheme = Some(Scheme::HTTPS);
  }
  if let (Some(path), Some(remove_path)) = (&uri_parts.path_and_query, remove_path) {
    if path.as_str().starts_with(remove_path) {
      uri_parts.path_and_query =
        Some(PathAndQuery::from_str(&path.as_str().replace(remove_path, "")).unwrap())
    }
  }

  if !uri.path().is_empty() {
    let mut new_path = uri.path().trim_end_matches("/").to_string();
    if let Some(path) = &uri_parts.path_and_query {
      new_path = format!("{new_path}{path}");
    }
    uri_parts.path_and_query = Some(PathAndQuery::from_str(&new_path).unwrap());
  }
  uri_parts.authority = uri.authority().cloned();
  *request.uri_mut() = Uri::from_parts(uri_parts).unwrap();

  request.headers_mut().insert(
    "host",
    HeaderValue::from_str(uri.authority().unwrap().as_str()).unwrap(),
  );
  if let Some(key) = key {
    request.headers_mut().insert(
      "authorization",
      HeaderValue::from_str(&format!("Bearer {key}")).unwrap(),
    );
  }

  request.headers_mut().remove("content-length");

  match client.request(request).await {
    Ok(resp) => resp.into_response(),
    Err(err) => (
      StatusCode::BAD_GATEWAY,
      format!("Reversed Proxy Error: {err:?}"),
    )
      .into_response(),
  }
}
