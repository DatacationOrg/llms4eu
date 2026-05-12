import uvicorn


def main() -> None:
    uvicorn.run("llms4eu.app:app", host="127.0.0.1", port=8080, reload=True)


if __name__ == "__main__":
    main()
