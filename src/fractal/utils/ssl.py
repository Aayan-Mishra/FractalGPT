from __future__ import annotations

import os


def configure_ssl_certificates() -> None:
    """Best-effort SSL certificate configuration.

    Some Python distributions (notably on macOS) can lack a usable default CA
    bundle for urllib-based HTTPS downloads. ESM checkpoints are downloaded via
    torch.hub/urllib, which respects SSL_CERT_FILE.

    We avoid disabling verification; instead we point at certifi's CA bundle when
    available and when the user hasn't already configured certs.
    """

    if os.environ.get("SSL_CERT_FILE") or os.environ.get("REQUESTS_CA_BUNDLE"):
        return

    try:
        import certifi  # type: ignore

        os.environ["SSL_CERT_FILE"] = certifi.where()
    except Exception:
        # If certifi isn't available, do nothing and let the caller raise.
        return
