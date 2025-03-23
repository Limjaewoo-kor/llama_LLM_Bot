# llama_LLM_Bot


임베딩 = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
응답모델 =  Ollama(model="mistral")

streamlit을 이용하여 type=['pdf', 'ppt', 'pptx', 'xls', 'xlsx', 'csv', 'txt', 'xml'])의 문서들을 업로드 받은 후 청킹하여 rag로 활용하여 답하는 챗봇입니다.



**
주석되어있는 
OpenAIEmbeddings(model="text-embedding-3-small")
model = ChatOpenAI(model="gpt-4o-mini")

위 모델을 이용하면 한글에 대한 정확도가 많이 상승하나 .env파일을 생성하여 gpt_api를 등록하고 오픈 api에 페이 충전후 사용하여야합니다.

<img width="1388" alt="image" src="https://github.com/user-attachments/assets/c6992f37-ab4c-4d9b-9e24-975c43214f80" />
