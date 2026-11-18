"""Add a readiness endpoint to the bundled Kev server."""
from kev import serve


@serve.app.get("/ready")
def ready():
    # serve.main() starts Uvicorn only after the selected checkpoint has loaded.
    return {"ready": hasattr(serve.app.state, "server")}


if __name__ == "__main__":
    serve.main()
