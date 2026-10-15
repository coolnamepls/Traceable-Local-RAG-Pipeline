#Docling converter creation
from docling.document_converter import DocumentConverter ,PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling_core.types.doc.document import DoclingDocument
import docling_core.types.doc.items as doc_items
import gc, json, os, torch


#Docling parsing
from docling_core.transforms.serializer.markdown import MarkdownParams
from docling_core.transforms.chunker.hierarchical_chunker import ChunkingDocSerializer,ChunkingSerializerProvider
from docling_core.transforms.chunker.hybrid_chunker import HybridChunker
from docling_core.transforms.chunker.tokenizer.base import BaseTokenizer
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer

#tokeniser and vlm
from transformers import AutoTokenizer, AutoProcessor, Qwen2_5_VLForConditionalGeneration, BitsAndBytesConfig
import PIL
from docling.chunking import DocChunk, DocMeta




#Helpers
##################################################

#turn document items into an json parsable dictionary
def item_to_json(item:doc_items.text.TextItem |doc_items.text.ListItem| doc_items.text.SectionHeaderItem | doc_items.table.table.TableItem|
                 doc_items.picture.picture.PictureItem, doc: DoclingDocument):
    
    if type(item) != doc_items.table.table.TableItem and type(item) != doc_items.picture.picture.PictureItem:
        textitem = item
        data = json.loads(textitem.model_dump_json(by_alias=True))
        pages = [p['page_no'] for p in data['prov']]
        data = {"self_ref": data['self_ref'],
                "parent": data['parent'],
                "label":data['label'],
                "text":data['text'],
                "page":pages}
    elif type(item) == doc_items.table.table.TableItem:
        data = json.loads(item.model_dump_json(by_alias=True))
        pages = [p['page_no'] for p in data['prov']]
        data = {"self_ref": data['self_ref'],
                "parent": data['parent'],
                "label":data['label'],
                "page":pages}      
    elif type(item) == doc_items.picture.picture.PictureItem:
        data = json.loads(doc.pictures[0].model_dump_json(by_alias=True))

        pages = [p['page_no'] for p in data['prov']]
        data = {"self_ref": data['self_ref'],
                "parent": data['parent'],
                "label":data['label'],
                "captions":data['captions'],
                "image":data['image'],
                "page":pages}
    else:
        data = -1
    return data

#loads VLM with appropriate prompt  for getting figure descriptions
def VLM(model:Qwen2_5_VLForConditionalGeneration,processor:AutoProcessor,image:doc_items.picture.picture.PictureItem, max_new_tokens: int, 
        doc: DoclingDocument, prompt: str = None):
    im = image.get_image(doc)
    cap = [t.resolve(doc).text for t in image.captions]
    prompt = f"""If the figure lacks context to be informative, ONLY output: 'N/A not enough context'. Else, if the figure is informative, describe the figure for a retrieval pipeline by 
    explaining what it shows, identify important elements, trends and relationships, axes, and provide a conclusion. Captions: {cap}""" if not prompt else prompt

    message = {"role":"user",
            "content":[{"type":"image", "image":im}, {"type":"text","text":prompt}]}
    formats = processor.apply_chat_template([message], tokenize =True, return_dict=True, return_tensors="pt", add_generation_prompt=True).to(model.device)

    with torch.no_grad():
        outs = model.generate(**formats, max_new_tokens = max_new_tokens)

    outs = processor.decode(outs[0][formats["input_ids"].shape[1]:], skip_special_tokens = True)
    return outs

# Gets heading of figures to preserve semantic meaning with other normal chunk sections
def get_headings(pic: doc_items.picture.picture.PictureItem, chunks: list):
    ref = pic.self_ref
    heading = None
    for c in chunks:
        dis = c.meta.doc_items
        for di in dis:
            if di.self_ref==ref:
                heading = c.meta.headings
                break
    return heading
    


#MAIN FUNCTIONS
#####################################

#Get a document object from docling after scanned file
def scanning_doc(file_scanned:str, converter: DocumentConverter):
    doc = converter.convert(file_scanned)
    doc = doc.document
    return doc






#saving infomration on scanned text into a json file
def save_scanned_text(doc:DoclingDocument, folder: str ):
    jsonl = os.path.join(folder,f"text_data.jsonl")
    with open(jsonl, "w", encoding = "utf-8") as f:
        i =0
        for t in doc.texts:
            i+=1
            ref = t.self_ref
            data = item_to_json(t, doc)
            f.write(json.dumps(data)+"\n")
            if i%100==0:
                print(data)    
                print(f"{ref} added")
    print("SCANNED TEXT ADDED!")
    print("="*70)



#saving informaiton of scanned tables into respective folders for the tables and their metadata 
def save_scanned_table(doc:DoclingDocument, table_folder: str):

    #set up folders
    table_meta_folder = os.path.join(table_folder,f"metadata")
    os.makedirs(name = table_meta_folder, exist_ok=True)
    table_md_folder = os.path.join(table_folder,f"markdown")
    os.makedirs(name = table_md_folder, exist_ok=True)
    
    for table in doc.tables:
        ref = table.self_ref
        name = "".join(ref.split("/")[1:])

        #Save metadata
        meta = item_to_json(table, doc)
        os.makedirs(name = table_meta_folder, exist_ok=True)
        with open(os.path.join(table_meta_folder, f"{name}_meta.json"),"w", encoding="utf-8") as j:
            json.dump(meta, j,  ensure_ascii=False, indent=2)

        #save table to markdown
        md_path = os.path.join(table_md_folder,f"{name}.md")
        data=table.export_to_markdown(doc)
        with open(md_path, "w", encoding="utf-8") as md:
            md.write(data)
        print(f"{ref} added")
    print("SCANNED TABLES ADDED!")
    print("="*70)


#saving informaiton of scanned figures into respective image for the tables and their metadata 
def save_scanned_figures(doc:DoclingDocument, image_folder: str):

    #Set up folders
    image_meta_folder = os.path.join(image_folder,f"metadata")
    image_figures_folder = os.path.join(image_folder,f"actual_image")
    os.makedirs(image_meta_folder, exist_ok=True)
    os.makedirs(image_figures_folder,exist_ok=True)
    i=0
    for pic_item in doc.pictures:
        i+=1
        ref = pic_item.self_ref
        name = "".join(ref.split(r"/")[1:])

        #Save metadata
        meta = item_to_json(pic_item, doc)
        with open(os.path.join(image_meta_folder, f"{name}_meta.json"),"w", encoding="utf-8") as j:
            json.dump(meta, j,  ensure_ascii=False, indent=2)

        #Save figure as png
        pic = pic_item.get_image(doc)
        fp = os.path.join(image_figures_folder,f"{name}.png")
        pic.save(fp)
        if i%2==0:
            print(f"{name} added!")
    print("SCANNED FIGURES ADDED!")

#The main function to automatically save all elements scanned files
def save_scanned_elements(doc:DoclingDocument, scanned_file:str, name_id:str):
    main_folder  = fr".\{scanned_file}_{name_id}_folder"

    #text
    folder = os.path.join(main_folder,"text")
    os.makedirs(name = folder, exist_ok=True)
    save_scanned_text(doc=doc, folder=folder)

    #table
    if doc.tables:
        table_folder  =os.path.join(main_folder,"tables")
        save_scanned_table(doc=doc, table_folder=table_folder)

    #Image
    if doc.pictures:
        image_folder  = os.path.join(main_folder,"figures")
        save_scanned_figures(doc=doc, image_folder=image_folder)







#Chunker, chunking
def chunking(doc: DoclingDocument, tokeniser: BaseTokenizer, serializer_provider: ChunkingSerializerProvider):

    chunker = HybridChunker(tokenizer=tokeniser, repeat_table_header=True, serializer_provider= serializer_provider)
    chunks = chunker.chunk(dl_doc=doc)
    chunks = list(chunks)
    return chunker, chunks

#creating chunks for figures, by adding their descriptions and metadata
def chunk_pic(doc: DoclingDocument, processor, chunks:list,model_vlm,max_new_tokens=900):

    if doc.pictures:

        pic_chunks = []
        for pic in doc.pictures:
            captions = "\n".join([t.resolve(doc).text for t in pic.captions])
            desc = VLM(model = model_vlm, processor=processor, image = pic, max_new_tokens=max_new_tokens,doc=doc)
            heading = get_headings(pic = pic, chunks=chunks)

            text = f"Caption: {captions}\n Figure details: {desc}"
            meta = DocMeta(doc_items = [pic], headings=heading, origin = doc.origin)
            p_c = DocChunk(text = text,meta=meta)
            pic_chunks.append(p_c)
            print(f"{pic.self_ref} Done!")
        
        return pic_chunks
    else:
        return []




#loop to save contents and metadata of each chunk into their respective folders

#Normal chunks save
def save_normal_chunks(chunks: list, main_folder: str, State: bool, chunker):
    N= len(chunks)
    for i in range(N):
        c=chunks[i]
        name = f"chunk{i}"
        chunk_folder = os.path.join(main_folder,name)
        os.makedirs(name=chunk_folder, exist_ok=True)


        #pages:
        start = c.meta.doc_items[0].prov[0].page_no
        end = c.meta.doc_items[-1].prov[-1].page_no
        pages = f"Page: {start}" if start==end else f"Page: {start}-{end}"

        #content
        text_folder = os.path.join(chunk_folder,"text")
        os.makedirs(name=text_folder, exist_ok=True)
        text_file = os.path.join(text_folder, f"{name}.txt")
        with open(text_file, "w", encoding = "utf-8") as f:
            text = pages+"\n\n"+chunker.contextualize(c)
            f.write(text)

        #meta
        meta_folder = os.path.join(chunk_folder,"metadata")
        os.makedirs(name=meta_folder, exist_ok=True)
        meta_file = os.path.join(meta_folder, f"{name}.json")
        with open(meta_file, "w", encoding = "utf-8") as f:
            json.dump(c.export_json_dict()['meta'],fp= f, ensure_ascii=False, indent=2)
        
        if i%20==0:
            print(f"{name} done!")
    State = True
    return State

#Figure chunk save
def save_figure_chunks(pic_chunks: list,chunks:list, main_folder: str, State: bool, chunker):
    if pic_chunks:
        if not State:
            print("Haven't saved normal Chunks!")
            return -1
        #figure chunks
        N2 = len(pic_chunks)
        N = len(chunks)
        
        for i in range(N2):
            
            c=pic_chunks[i]
            name = f"chunk{N+i}"
            chunk_folder = os.path.join(main_folder,name)
            os.makedirs(name=chunk_folder, exist_ok=True)

            #pages:
            start = c.meta.doc_items[0].prov[0].page_no
            end = c.meta.doc_items[-1].prov[-1].page_no
            pages = f"Page: {start}" if start==end else f"Page: {start}-{end}"

            #content
            text_folder = os.path.join(chunk_folder,"text")
            os.makedirs(name=text_folder, exist_ok=True)
            text_file = os.path.join(text_folder, f"{name}.txt")
            with open(text_file, "w", encoding = "utf-8") as f:
                text = pages+"\n\n"+chunker.contextualize(c)
                #json.dump(c.export_json_dict()['text'],fp= f, ensure_ascii=False, indent=2)
                f.write(text)

            #meta
            meta_folder = os.path.join(chunk_folder,"metadata")
            os.makedirs(name=meta_folder, exist_ok=True)
            meta_file = os.path.join(meta_folder, f"{name}.json")
            with open(meta_file, "w", encoding = "utf-8") as f:
                json.dump(c.export_json_dict()['meta'],fp= f, ensure_ascii=False, indent=2)

            if i%2==0:
                print(f"{name} done!")   


#ALL chunk save
def save_chunks(chunks: list, pic_chunks:list, file_scanned:str, chunker, name_id):
    main_folder = os.path.join(f".\{file_scanned}_{name_id}_folder", "chunks")
    os.makedirs(name=main_folder, exist_ok=True)
    total_chunks = chunks+pic_chunks


    State = False
    State = save_normal_chunks(chunks = chunks, main_folder=main_folder, State=State, chunker=chunker)
    save_figure_chunks(pic_chunks=pic_chunks, chunks=chunks, main_folder=main_folder,State=State, chunker=chunker)

    data = []
    data_text =[]
    for c in total_chunks:
        d = c.export_json_dict()
        start = c.meta.doc_items[0].prov[0].page_no
        end = c.meta.doc_items[-1].prov[-1].page_no
        pages = f"Page: {start}" if start==end else f"Page: {start}-{end}"
        text = pages+"\n\n"+d['text']
        d['text'] = text
        data.append(d)
        data_text.append(d['text'])
    path = os.path.join(main_folder,"total_chunks.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f,indent=4)
    print("ALL CHUNKS SAVED!!!")
    return data, data_text


##########################################################################
#Implementation functions

def YorN(que:str):
    while True:
        ans = input(que)
        if ans=="Y":
            return True
        elif ans=="N":
            return False
        else:
            print("Y/N only")



def docling_scan_chunk_save(file_scanned, embed_model_id, vlm_id, name_id, device, skip=False):
    main_folder = rf".\{file_scanned}_{name_id}_folder"
    if skip:
        if os.path.exists(main_folder):
            return None, None
        else:
            print("No records found!")

    

    if os.path.exists(main_folder):

        if not YorN("This folder already exist, do you want to replace it (Y/N):"):
            return None, None


    #Set up:
    #docling pdf converter
    pipeline = PdfPipelineOptions(do_table_structure=True, generate_page_images=False, do_ocr=True, generate_picture_images=True)
    converter = DocumentConverter(format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline)})
    print("LOADED CONVERTER")



    ########################################################################################

    #scanned documents
    doc = scanning_doc(file_scanned = file_scanned,converter=converter)

    import gc
    del converter 
    gc.collect()
    torch.cuda.empty_cache()

    #VLM for getting description of figures
    processor = AutoProcessor.from_pretrained(vlm_id)
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
    model_vlm = Qwen2_5_VLForConditionalGeneration.from_pretrained(pretrained_model_name_or_path=vlm_id, quantization_config = quant, device_map = device)
    print("LOADED VLM")


    #save scanned elements
    save_scanned_elements(doc=doc, scanned_file=file_scanned, name_id=name_id)

    #chunker, chunking
    tokeniser_raw = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=embed_model_id)
    tokeniser: BaseTokenizer = HuggingFaceTokenizer(tokenizer=tokeniser_raw, max_tokens = 720)
    class ImgPlaceholderSerializerProvider(ChunkingSerializerProvider):
        def get_serializer(self, doc):
            return ChunkingDocSerializer(
                doc=doc,
                params=MarkdownParams(
                    image_placeholder="<!-- image -->",
                ),
            )
    chunker, chunks = chunking(doc=doc, tokeniser= tokeniser, serializer_provider= ImgPlaceholderSerializerProvider())

    #providing desc for figures, then saving the chunks into repsective folder, with their data and metadata
    pic_chunks = chunk_pic(doc=doc, processor = processor, max_new_tokens=900,model_vlm=model_vlm, chunks=chunks)
    total_chunks, text_to_embed = save_chunks(chunks = chunks, pic_chunks=pic_chunks, file_scanned=file_scanned, chunker=chunker, name_id=name_id)

    import gc
    del model_vlm
    gc.collect()
    torch.cuda.empty_cache()
    return total_chunks, text_to_embed






####################################################################
if __name__ == "__main__":

    #embed_model_id for hybrid chunking in docling
    embed_model_id = "Qwen/Qwen3-Embedding-0.6B"
    vlm_id = 'Qwen/Qwen2.5-VL-3B-Instruct'
    file_scanned = "BDCC-09-00093.pdf"

    #setting up folder name
    embed_m_name = embed_model_id.split("/")[-1]
    vlm_name = vlm_id.split("/")[-1]
    name_id = embed_m_name+vlm_name+"_docling"
    total_chunks, text_to_embed = docling_scan_chunk_save(file_scanned=file_scanned, embed_model_id=embed_model_id,
                            vlm_id=vlm_id, name_id=name_id,skip=True)
    print(type(total_chunks))
    print(type(text_to_embed))