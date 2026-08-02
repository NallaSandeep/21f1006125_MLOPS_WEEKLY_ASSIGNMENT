from fastapi import FastAPI, Request, HTTPException, Response, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import mlflow.pyfunc
import pandas as pd
import time
import logging
import json
from datetime import datetime, timezone

from contextlib import asynccontextmanager

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.cloud_trace import CloudTraceSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

MODEL_PATH = "/app/model"

app_state = {
    "is_ready": False,
    "is_alive": True,
}

model = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model

    # Startup
    model = mlflow.pyfunc.load_model(MODEL_PATH)
    app_state["is_ready"] = True

    yield

    # Shutdown
    app_state["is_ready"] = False

# Setup Tracer
tracer_provider = TracerProvider()
trace.set_tracer_provider(tracer_provider)
tracer = trace.get_tracer(__name__)
span_processor = BatchSpanProcessor(CloudTraceSpanExporter())
trace.get_tracer_provider().add_span_processor(span_processor)

# Setup structured logging
logger = logging.getLogger("iris-prediction-service")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()

STANDARD_LOG_ATTRIBUTES = {
    "name",
    "msg",
    "args",
    "levelname",
    "levelno",
    "pathname",
    "filename",
    "module",
    "exc_info",
    "exc_text",
    "stack_info",
    "lineno",
    "funcName",
    "created",
    "msecs",
    "relativeCreated",
    "thread",
    "threadName",
    "processName",
    "process",
    "taskName",
}

class JsonFormatter(logging.Formatter):

    def format(self, record):
        log_entry = {
            "severity": record.levelname,
            "message": record.getMessage(),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        # -------------------------------------------------
        # Add OpenTelemetry trace information
        # -------------------------------------------------

        span = trace.get_current_span()
        span_context = span.get_span_context()

        if span_context.is_valid:
            log_entry["trace_id"] = format(
                span_context.trace_id,
                "032x"
            )

            log_entry["span_id"] = format(
                span_context.span_id,
                "016x"
            )

        # -------------------------------------------------
        # Add fields passed through extra={}
        # -------------------------------------------------

        for key, value in record.__dict__.items():
            if key not in STANDARD_LOG_ATTRIBUTES:
                log_entry[key] = value

        # Include exception details
        if record.exc_info:
            log_entry["exception"] = self.formatException(
                record.exc_info
            )

        return json.dumps(
            log_entry,
            default=str
        )

handler.setFormatter(JsonFormatter())
logger.addHandler(handler)

app = FastAPI(title="Iris Prediction API", lifespan=lifespan)

FastAPIInstrumentor.instrument_app(
    app,
    tracer_provider=tracer_provider
)

class IrisRequest(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float


@app.get("/")
def health():
    return {"status": "healthy"}

@app.get("/live_check", tags=["Probe"])
async def liveness_probe():
    if app_state["is_alive"]:
        return {"status": "alive"}
    return Response(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR)

@app.get("/ready_check", tags=["Probe"])
async def readiness_probe():
    if app_state["is_ready"]:
        return {"status": "ready"}
    return Response(status_code=status.HTTP_503_SERVICE_UNAVAILABLE)

@app.middleware("http")
async def access_log_middleware(request: Request, call_next):
    start_time = time.perf_counter()

    # Capture request body
    request_body_bytes = await request.body()

    try:
        request_body = (
            json.loads(request_body_bytes)
            if request_body_bytes
            else None
        )
    except (json.JSONDecodeError, UnicodeDecodeError):
        request_body = None

    # Process request
    response = await call_next(request)

    # Capture response body
    response_body_bytes = b""

    async for chunk in response.body_iterator:
        response_body_bytes += chunk

    try:
        response_body = (
            json.loads(response_body_bytes)
            if response_body_bytes
            else None
        )
    except (json.JSONDecodeError, UnicodeDecodeError):
        response_body = None

    duration_ms = round(
        (time.perf_counter() - start_time) * 1000,
        2
    )

    client_ip = (
        request.client.host
        if request.client
        else None
    )

    logger.info(
        "#Host: %s #AccessLog #HttpMethod: %s #URL: %s #Time: %.2f ms",
        client_ip,
        request.method,
        request.url.path,
        duration_ms,
        extra={
            "http_method": request.method,
            "http_path": request.url.path,
            "status_code": response.status_code,
            "duration_ms": duration_ms,
            "client_ip": client_ip,
            "request_body": request_body,
            "response_body": response_body,
        },
    )

    # We consumed body_iterator above, so reconstruct response
    new_response = Response(
        content=response_body_bytes,
        status_code=response.status_code,
        headers=dict(response.headers),
        media_type=response.media_type,
    )

    new_response.headers["X-Process-Time-ms"] = str(duration_ms)

    return new_response

@app.exception_handler(Exception)
async def exception_handler(request: Request, exc: Exception):
    span = trace.get_current_span()
    trace_id = format(span.get_span_context().trace_id, "032x")
    logger.exception(json.dumps({
        "event": "unhandled_exception",
        "trace_id": trace_id,
        "path": str(request.url),
        "error": str(exc)
    }))
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal Server Error", "trace_id": trace_id},
    )

@app.post("/predict")
def predict(request: IrisRequest):
    with tracer.start_as_current_span("model_inference") as span:
        span_context = span.get_span_context()
        span_id = (
            format(span_context.span_id, "016x")
            if span_context.is_valid
            else None
        )
        input_df = pd.DataFrame([request.model_dump()])

        prediction = model.predict(input_df)[0]

        response = {
            "predicted_class": prediction,
            "span_id": span_id,
        }

        logger.info("Predictions response %s", prediction, extra=response)
        return response
