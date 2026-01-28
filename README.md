Estudo Autônomo de RAG
Autonomous RAG Study Project

========================================================================================================================================================================================================


PT-BR:

LLMs: Ollama (Local) / OpenAI
Linguagem: Python

Descrição:
Este projeto consiste em um estudo autônomo sobre Retrieval-Augmented Generation (RAG).
O sistema utiliza chunks de texto simples e ficcionais como base de conhecimento. A entrada do usuário é transformada em embeddings e comparada semanticamente com os embeddings das chunks armazenadas.
Quando o score de similaridade atinge ou supera o valor mínimo definido, a chunk mais relevante é enviada como contexto para a LLM, que é instruída a responder exclusivamente com base nesse conteúdo.
Quando o score não atinge o mínimo (a informação não está presente nas chunks), o sistema libera a LLM para utilizar conhecimento geral a fim de responder à pergunta do usuário.
O projeto conta com duas LLMs para fins de estudo e testes:
Uma LLM local (Ollama), atualmente comentada no código.
Uma LLM em nuvem (OpenAI), utilizada como padrão.
Essa arquitetura permite alternar facilmente entre modelos locais e remotos mantendo a mesma lógica de RAG.


========================================================================================================================================================================================================


🇺🇸 English
Autonomous RAG Study Project

LLMs: Ollama (Local) / OpenAI
Language: Python

Description:
This project is an autonomous study focused on Retrieval-Augmented Generation (RAG).
The system uses simple, fictional text chunks as its knowledge base. User input is embedded and semantically compared against the stored chunk embeddings.
When the similarity score meets or exceeds the defined minimum threshold, the most relevant chunk is provided as context to the LLM, which is instructed to answer strictly based on that context.
When the score does not reach the minimum threshold (the information is not present in the chunks), the system allows the LLM to use its general knowledge to generate a response.
The project supports two LLM backends for experimentation purposes:
A local LLM (Ollama), currently commented out in the code.
A cloud-based LLM (OpenAI), enabled by default.
This architecture allows easy switching between local and remote models while preserving the same RAG logic.
