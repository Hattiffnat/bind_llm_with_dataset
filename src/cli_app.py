from typing import Any, List

import pandas
from kaggle.api.kaggle_api_extended import KaggleApi
from llama_cpp import Iterator, Llama
from RestrictedPython import safe_globals

DATASET_NAME = "shohinurpervezshohan/freelancer-earnings-and-job-trends"
DATASET_FILEPATH = f"dataset/freelancer_earnings_bd.csv"


def llm_response_to_str(response: Any) -> str:
    """Преобразует ответ от модели Llama в обычную строку."""
    result = []
    if isinstance(response, Iterator):
        [result.extend(x["choices"]) for x in response]
    else:
        result.extend(response["choices"])

    return "\n".join(map(lambda x: x["text"], result))


def build_prompt_with_history(history: List, new_prompt: str) -> str:
    """Собирает полный prompt, включая историю диалога."""
    history.append({"role": "user", "content": new_prompt})

    prompt = ""
    for msg in history:
        match msg["role"]:
            case "system":
                prompt += f"<s>[INST] {msg['content']} [/INST]\n"
            case "user":
                prompt += f"<s>[INST] {msg['content']} [/INST]\n"
            case "assistant":
                prompt += f"{msg['content']}\n"

    # print(f"PROMPT: {prompt} END")

    return prompt


def main():
    kaggle_api = KaggleApi()
    kaggle_api.authenticate()

    kaggle_api.dataset_download_files(dataset=DATASET_NAME, path="dataset", unzip=True)

    df: pandas.DataFrame = pandas.read_csv(DATASET_FILEPATH)

    llm = Llama.from_pretrained(
        repo_id="TheBloke/CodeLlama-13B-Instruct-GGUF",
        filename="codellama-13b-instruct.Q2_K.gguf",
        local_dir="models",
    )

    while user_question := input("Ваш вопрос: ").strip():
        history = []

        new_prompt = build_prompt_with_history(
            history,
            (
                f"Answer only code. No comments. "
                f"Write one-liner to extract data from df that look like this:\n{df.head(1)}\n"
                f'The extracted data must allow answering the question "{user_question}". '
            ),
        )

        response = llm(new_prompt, temperature=0.3, max_tokens=1024)
        # print(response)
        response_as_str = llm_response_to_str(response)

        history.append(
            {"role": "assistant", "content": response_as_str},
        )

        result = eval(response_as_str, safe_globals | {"df": df})

        new_prompt = f'Write a conclusion based on the result obtained "{result}"'
        print(
            llm_response_to_str(
                llm(
                    build_prompt_with_history(history, new_prompt),
                    temperature=0.3,
                    max_tokens=1024,
                )
            )
        )


if __name__ == "__main__":
    main()
