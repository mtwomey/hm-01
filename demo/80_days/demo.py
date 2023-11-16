import sys
import os
from haystack.nodes import AnswerParser, EmbeddingRetriever, PromptNode, PromptTemplate
from haystack.pipelines import Pipeline
from mongo_haystack import MongoAtlasDocumentStore
import warnings

# Ignore all warnings
warnings.filterwarnings("ignore")  # Supress the pytorch warning

openapi_key = os.getenv("OPENAI_KEY")

mongo_atlas_username = os.getenv("MONGO_ATLAS_USERNAME")
mongo_atlas_password = os.getenv("MONGO_ATLAS_PASSWORD")
mongo_atlas_host = os.getenv("MONGO_ATLAS_HOST")
mongo_atlas_database = os.getenv("MONGO_ATLAS_DATABASE")
mongo_atlas_collection = "80_days"
mongo_atlas_connection_params = {"retryWrites": "true", "w": "majority"}
mongo_atlas_params_string = "&".join([f"{key}={value}" for key, value in mongo_atlas_connection_params.items()])
mongo_atlas_connection_string = (
    f"mongodb+srv://{mongo_atlas_username}:{mongo_atlas_password}@{mongo_atlas_host}/?{mongo_atlas_params_string}"
)

embedding_dim = 768
embedding_model = "sentence-transformers/all-mpnet-base-v2"

# Provide option for a different sentance transformer with 1024 dim embeddings
if len(sys.argv) > 1:
    match sys.argv[1]:
        case "1024":
            embedding_dim = 1024
            embedding_model = "BAAI/bge-large-en-v1.5"
            print(f"\nWill use {embedding_model} transformer with dim = {embedding_dim}.\n")

document_store = MongoAtlasDocumentStore(
    mongo_connection_string=mongo_atlas_connection_string,
    database_name=mongo_atlas_database,
    collection_name=mongo_atlas_collection,
    embedding_dim=embedding_dim,
    progress_bar=False,
)

retriever = EmbeddingRetriever(
    document_store=document_store,
    embedding_model=embedding_model,
    model_format="sentence_transformers",
    top_k=10,
    progress_bar=False,
)


def main():
    print("\nWhat would you like to know about the book 'Around the World in Eighty Days' by Jules Verne?")
    while True:
        user_question = input("\nQuestion: ")
        if user_question.lower() in ["exit", "quit"]:
            break
        response = get_response(user_question)
        print(f"\n{response}\n")


def get_response(user_question):
    rag_prompt = PromptTemplate(
        prompt="""Synthesize a comprehensive answer from the following text for the given question.
                  Provide a clear and concise response that summarizes the key points and information presented in the text.
                  Your answer should be in your own words and be no longer than 50 words.
                  \n\n Related text: {join(documents)} \n\n Question: {query} \n\n Answer:""",
        output_parser=AnswerParser(),
    )
    prompt_node = PromptNode(
        model_name_or_path="gpt-3.5-turbo-instruct",
        api_key=openapi_key,
        default_prompt_template=rag_prompt,
        max_length=150,
    )
    pipe = Pipeline()
    pipe.add_node(component=retriever, name="retriever", inputs=["Query"])
    pipe.add_node(component=prompt_node, name="prompt_node", inputs=["retriever"])

    output = pipe.run(query=user_question)
    return output["answers"][0].answer


if __name__ == "__main__":
    main()
