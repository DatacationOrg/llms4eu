from pathlib import Path

import uvicorn

from src.shared.env import load_yaml


CONFIG = load_yaml(Path(__file__).resolve().parents[1] / "config.yaml")


def main() -> None:
    uvicorn.run(
        "src.scraping.web.app:app",
        host=CONFIG["web_host"],
        port=CONFIG["web_port"],
        reload=True,
    )


if __name__ == "__main__":
    main()
