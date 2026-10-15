import os, json, shutil, gc, torch

#Embedding

from sentence_transformers import SentenceTransformer
#embed
def embedding(text_to_embed: list, model=None, batch_size=8, device="cuda", skip=False):
    if skip:
        return
    
    embed_model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B", device = device) if not model else model
    batch_size = 8 if not batch_size else batch_size

    embeddings = embed_model.encode(
        text_to_embed,
        batch_size = batch_size,
        show_progress_bar = True,
        normalize_embeddings=True
    )
    print("EMBEDDING DONE!")

    if embed_model:
        del embed_model
        gc.collect()
        torch.cuda.empty_cache()
    return embeddings
#################################################################
#create vec db and store embeddings + metadata in collection

#Set up database
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
from qdrant_client.models import VectorParams, Distance

#Create empty database collection

def setup_qdrant_vecDB(embeddings, total_chunks, main_folder: str, skip=False):
    if skip:
        return

    
    db_path = os.path.join(main_folder,"qdrant_db")
    client = QdrantClient(path=db_path)
    collectionp = os.path.join(os.path.join(db_path,"collection"),"doc_chunks")

    #delete existing collection if there is one to replace with new collection
    if os.path.exists(collectionp):
        client.delete_collection(collection_name='doc_chunks')

    s = embeddings.shape[1]
    vec_params = VectorParams(size = s, distance = Distance.COSINE)
    client.create_collection(
        collection_name="doc_chunks",
        vectors_config=vec_params     
    )
    print("VECTOR DATABASE SETUP")

    #Creating appropriate points for uploading, then upload to the collection of the db
    points = []
    for i, (c, v) in enumerate(zip(total_chunks, embeddings)):
        info = {
            "id":i,
            "vector":v,
            "payload":{
                "text":c['text'],
                "metadata":c['meta']
            }
        }
        p = PointStruct(**info)
        points.append(p)

    client.upload_points(collection_name="doc_chunks",points=points)
    print("ADDED POINTS TO VECTOR DATABASE!")

    client.close()

#########################################################################################
if __name__ == "__main__":
    embed_model_id = "Qwen/Qwen3-Embedding-0.6B"
    vlm_id = 'Qwen/Qwen2.5-VL-3B-Instruct'
    file_scanned = "BDCC-09-00093.pdf"
    embed_m_name = embed_model_id.split("/")[-1]

    vlm_name = vlm_id.split("/")[-1]
    name_id = embed_m_name+vlm_name+"_docling"
    main_folder = rf"./{file_scanned}_{name_id}_folder"
    chunk_folder = os.path.join(main_folder,"chunks")
    print("\n")
    print(main_folder)
    print("\n\n")

    if not os.path.exists(main_folder):
        print(f"NO {main_folder} FOLDER FOUND!!")

    json_p = os.path.join(chunk_folder,"total_chunks.json")
    with open(json_p, "r",encoding="utf-8") as j:
        total_chunks = json.load(j)

    text_to_embed = [c['text'] for c in total_chunks]

    # Embedding
    embedding = embedding(text_to_embed=text_to_embed, model = None)
    

    setup_qdrant_vecDB(embeddings=embedding, main_folder=main_folder)
