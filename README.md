# Open Source knowledge Bot

## Solution Approach

For any textual knowledge base, first text snippets have to be extracted from the knowledge base. An embedding model is useed to create a vector store representing the semantic content of the snippets. A similarity search is performed from vector store according a question. After extracting the snippets, a prompt is enegineerd and answer is generated using the LLM generation model. The prompt can be tuned based on the specific LLM used.

## ChatGPT and langchain

### OPtions

- Embeddings: In the example, I OpenAI Embeddings (you may consider Cohere Embeddings, and HuggingFaceEmbeddings from specific models).
- TextSplitter: Character Text Splitter in the example where the text is split by a single character.
- VectorStore: We used Chroma as our vector database where we store our embedded text vectors. Other popular options are FAISS, Mulvus, and Pinecone.
- Retrievers: 
    - [DocArrayInMemorySearch](https://python.langchain.com/docs/integrations/vectorstores/docarray_in_memory) from Lamgchain is used, since we are doing some trials with few chunks of code
    - Ther also examples with Chroma returns . To retrieve text, there are two search types you can choose: search_type: “similarity” or “mmr”. search_type="similarity" uses similarity search in the retriever object where it selects text chunk vectors that are most similar to the question vector. search_type="mmr" uses the maximum marginal relevance search where it optimizes for similarity to query AND diversity among selected documents.

### Chain Type
The default chain_type="stuff" uses ALL of the text from the documents in the prompt. It may exceeds the token limit and causes rate-limiting errors. Other options:

- map_reduce: It separates texts into batches (as an example, you can define batch size in llm=OpenAI(batch_size=5)), feeds each batch with the question to LLM separately, and comes up with the final answer based on the answers from each batch.
- refine : It separates texts into batches, feeds the first batch to LLM, and feeds the answer and the second batch to LLM. It refines the answer by going through all the batches.
- map-rerank: It separates texts into batches, feeds each batch to LLM, returns a score of how fully it answers the question, and comes up with the final answer based on the high-scored answers from each batch.

One issue with using ALL of the text is that it can be very costly because you are feeding all the texts to OpenAI API and the API is charged by the number of tokens. A better solution is to retrieve relevant text chunks first and only use the relevant text chunks in the language model.

RetrievalQA chain actually uses load_qa_chain under the hood. We retrieve the most relevant chunk of text and feed those to the language model.

## Open Source LLMs

[Leaderboard LLMs](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard)

Under constructio, use FLAN:

- FlanT5 Models : FlanT5 is text2text generator that is finetuned on several tasks like summarisation and answering questions. It uses the encode-decoder architecture of transformers. The model is Apache 2.0 licensed, which can be used commercially.
- FastChatT5 3b Model : It's a FlanT5-based chat model trained by fine tuning FlanT5 on user chats from ChatGPT. The model is Apache 2.0 licensed.
- Falcon7b Model : Falcon7b is a smaller version of Falcon-40b, which is a text generator model (decoder-only model). - Falcon-40B is currently the best open-source model on the OpenLLM Leaderboard. One major reason for its high performance is its training with high-quality data. 

For CPU based systems: SBERT for the embedding model and FLANT5-Base for the generation model. 

## Embeddings

We use the following Open Source models in the codebase:

INSTRUCTOR XL : Instructor xl is an instruction-finetuned text embedding model that can generate embeddings tailored for any task instruction. The instruction for embedding text snippets is "Represent the document for retrieval:". The instruction for embedding user questions is "Represent the question for retrieving supporting documents:"
SBERT : SBERT maps sentences and paragraphs to vectors using a BERT-like model. It's a good start when we’re prototyping our application.

[LeaderBoard Embeddings](https://huggingface.co/spaces/mteb/leaderboard)

## Implementation

To employ these models, we use Hugging Face pipelines, which simplify the process of loading the models and using them for inference. 

For encoder-decoder models like FlanT5, the pipeline’s task is ”text2text-generation”. 
The auto device map feature assists in efficiently loading the language model (LLM) by utilizing GPU memory. If the entire model cannot fit in the GPU memory, some layers are loaded onto the CPU memory instead. If the model still cannot fit completely, the remaining weights are stored in disk space until needed.
Loading in 8-bit quantizes the LLM and can lower the memory requirements by half.
The creation of the models is governed by the configuration settings and is handled by the create_sbert_mpnet() and create_flan_t5_base() functions, respectively.
