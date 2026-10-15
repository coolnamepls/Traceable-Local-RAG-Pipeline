from sentence_transformers import SentenceTransformer, CrossEncoder
from transformers import AutoTokenizer, AutoModelForCausalLM,  BitsAndBytesConfig


from qdrant_client import QdrantClient
import torch, gc, os, json, time

from docling_parser import docling_scan_chunk_save
from tem_database_setup import embedding, setup_qdrant_vecDB

def name_id_setup(embed_model_id:str, vlm_id:str):
    #setting up folder name
    embed_m_name = embed_model_id.split("/")[-1]
    vlm_name = vlm_id.split("/")[-1]
    name_id = embed_m_name+vlm_name+"_docling"
    return name_id


def LLM(model,tokeniser, prompt, max_new_tokens = None,additional=None):

    #creating the prompt with query, retrieved info and additional instructions
    full_text = f"{prompt}"
    if additional:
        full_text = additional + "\n" + full_text 

    #form the message as input for llm, reformat to chat template for llm
    message = {"role":"user",
               "content":full_text}

    formated = tokeniser.apply_chat_template(
        [message],
        tokenize = False,
        add_generation_prompt=True,
        enable_thinking = False 
        )

    #set input and set max tokens
    inputs = tokeniser(formated, return_tensors="pt").to(model.device)

    #output from answer and set max tokens
    outputs = model.generate(**inputs, max_new_tokens = max_new_tokens) if max_new_tokens else model.generate(**inputs)

    #print text start from answer and remove special token
    repsonse = tokeniser.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)    
    return repsonse


def RAG_LLM(model,tokeniser, query,retrieved, max_new_tokens = None,additional=None):

    #creating the prompt with query, retrieved info and additional instructions
    full_text = f"query: {query}"+ "\n"+f"Retrieved data:{retrieved}"
    if additional:
        full_text = additional + "\n" + full_text 

    #form the message as input for llm, reformat to chat template for llm
    message = {"role":"user",
               "content":full_text}

    formated = tokeniser.apply_chat_template(
        [message],
        tokenize = False,
        add_generation_prompt=True,
        enable_thinking = False 
        )

    #set input and set max tokens
    inputs = tokeniser(formated, return_tensors="pt").to(model.device)

    #output from answer and set max tokens
    outputs = model.generate(**inputs, max_new_tokens = max_new_tokens) if max_new_tokens else model.generate(**inputs)

    #print text start from answer and remove special token
    repsonse = tokeniser.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)    
    return repsonse, full_text


def RAG_QNA(main_folder,out_name,model, tokeniser, embed_model, reranker, limit,query,device = "cuda" ,max_new_tokens=2048, additional=None):


    #embed query and deletes model
    eQ  = embed_model.encode(inputs=query, device = device, normalize_embeddings=True)
    del embed_model
    gc.collect()
    torch.cuda.empty_cache()
    print("EMBEDED!")

    #load collection and retrieve answers, number of retrievals: 2 default
    db_path = rf"./{main_folder}/qdrant_db"
    client = QdrantClient(path=db_path)
    client.get_collection("doc_chunks")
    Retrieve = client.query_points(collection_name="doc_chunks",query=eQ.tolist(),limit = limit)
    points = Retrieve.points
    client.close()
    print("RETRIEVED!")

    #Reranking retreived chunks and save chunks retrieved
    name = out_name
    output_path = os.path.join(os.path.join(main_folder, "results"), name)
    
    os.makedirs(name = output_path, exist_ok = True)
    qp_pair = [(query, p.payload['text']) for p in points]
    _scores = reranker.predict(qp_pair)
    ranked = sorted(zip(points,_scores), key = lambda x: x[1], reverse= True)
    del reranker
    gc.collect()
    torch.cuda.empty_cache()

    print("RANKED!")
    text = []
    i=0
    for p,s  in ranked:
        i+=1
        ids = p.id
        score = str(s)
        texts = f"CHUNK{ids}\nScore: {score}\n\n"+p.payload['text']
        path =  os.path.join(output_path, f"{i}_chunk{ids}.txt")
        with open(path, "w", encoding = "utf-8") as f:
            f.write(texts)
        text.append(p.payload['text'])
    #text = [(r.payload)['text'] for r in points]
    
    #response from LLM
    response, full_text = RAG_LLM(model = model, tokeniser=tokeniser, query=query, retrieved=text, max_new_tokens=max_new_tokens, additional=additional)

    #Saving prompt
    prompt_path = os.path.join(output_path,f"{name}_prompt.txt")
    with open(prompt_path, "w", encoding = "utf-8") as f:
        f.write(full_text)
    
    #saving response in markdown
    output_path = os.path.join(output_path, f"{name}.md")
    with open(output_path, "w", encoding="utf-8") as md:
        md.write(response)
    print(f"Response added at {output_path}")
    return response, full_text


#################################################################################
#test entire RAG process

if __name__ == "__main__":
    

    #embed_model_id for hybrid chunking in docling, skip processes by default, device = cuda by default
    embed_model_id = "Qwen/Qwen3-Embedding-0.6B"
    vlm_id = 'Qwen/Qwen2.5-VL-3B-Instruct'
    file_scanned = "AFRC_Audit Focus for 2025 year-end audits (FS Sector)_3 Oct 2025.pdf"
    name_id = name_id_setup(embed_model_id=embed_model_id,vlm_id=vlm_id)
    main_folder = rf"./{file_scanned}_{name_id}_folder"
    skip = True
    device = "cuda"

    #add device option
    total_chunks, text_to_embed = docling_scan_chunk_save(file_scanned=file_scanned, embed_model_id=embed_model_id,
                            vlm_id=vlm_id, name_id=name_id, device=device,skip=skip)
    
    '''
    with open(f"{main_folder}/chunks/total_chunks.json", "r", encoding="utf-8") as j:
        total_chunks = json.load(j)
    text_to_embed = [c['text'] for c in total_chunks]
    '''
    embeddings = embedding(text_to_embed=text_to_embed, model=None, batch_size=8, device=device, skip=skip)
    setup_qdrant_vecDB(embeddings=embeddings, total_chunks = total_chunks, main_folder=main_folder, skip=skip)
    

    #load embedding model,llm model, tokeniser, quantisation methods
    chunk_limit = 5
    embed_model_id = "Qwen/Qwen3-Embedding-0.6B"
    Cross_encoder= "Qwen/Qwen3-Reranker-0.6B"
    model_id = "Qwen/Qwen3-4B"
    embed_model = SentenceTransformer(model_name_or_path=embed_model_id,device=device)
    quant_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16, bnb_4bit_quant_type="nf4")
    tokeniser = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_id)
    model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=model_id, device_map = device, quantization_config = quant_config)
    reranker = CrossEncoder(model_name_or_path=Cross_encoder, device = device, activation_fn=torch.nn.Sigmoid())

    #assume we have a collection, we have a query, we have a model, we have a tokeniser
    query = "What is the audit focus of 2025 and what should audit firms pay attention to?"
    prompt = "Give me a report listing out the focus of auditing and what should audit firms do correpspondingly"
    response, prompt = RAG_QNA(main_folder = main_folder,out_name = "test",model = model, tokeniser = tokeniser, embed_model = embed_model, 
                       reranker=reranker,limit = chunk_limit ,query = query,device = "cuda" ,max_new_tokens=2048, additional=prompt)
'''

    #embed query and deletes model
    eQ  = embed_model.encode(inputs=query, device = device, normalize_embeddings=True)
    del embed_model
    gc.collect()
    torch.cuda.empty_cache()
    print("EMBEDED!")

    #load collection and retrieve answers, number of retrievals: 2 default
    db_path = rf"./{main_folder}/qdrant_db"
    client = QdrantClient(path=db_path)
    client.get_collection("doc_chunks")
    Retrieve = client.query_points(collection_name="doc_chunks",query=eQ.tolist(),limit = 100)
    points = Retrieve.points
    client.close()
    print("RETRIEVED!")

    #Reranking retreived chunks and save chunks retrieved
    name = "test_response0"
    output_path = os.path.join(os.path.join(main_folder, "results"), name)
    
    os.makedirs(name = output_path, exist_ok = True)
    qp_pair = [(query, p.payload['text']) for p in points]
    _scores = reranker.predict(qp_pair)
    ranked = sorted(zip(points,_scores), key = lambda x: x[1], reverse= True)

    print("RANKED!")
    text = []
    i=0
    for p,s  in ranked:
        i+=1
        ids = p.id
        score = str(s)
        texts = f"CHUNK{ids}\nScore: {score}\n\n"+p.payload['text']
        path =  os.path.join(output_path, f"{i}_chunk{ids}.txt")
        with open(path, "w", encoding = "utf-8") as f:
            f.write(texts)
        text.append(p.payload['text'])
    #text = [(r.payload)['text'] for r in points]
    
    #response from LLM
    response = RAG_LLM(model = model, tokeniser=tokeniser, query=query, retrieved=text, max_new_tokens=2048, additional=None)


    #saving response in markdown
    output_path = os.path.join(output_path, f"{name}.md")
    with open(output_path, "w", encoding="utf-8") as md:
        md.write(response)
    print(f"Response added at {output_path}")
    #print(response)
 '''   