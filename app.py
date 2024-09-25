import streamlit as st
import base64
from huggingface_hub import notebook_login
from byaldi import RAGMultiModalModel
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from PIL import Image
from io import BytesIO
import torch
import re

@st.cache_resource
def load_models():
    RAG = RAGMultiModalModel.from_pretrained("vidore/colpali", verbose=10)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2-VL-2B-Instruct",
        torch_dtype=torch.float16,
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")
    return RAG, model, processor

RAG, model, processor = load_models()

st.title("Multimodal Image Search and Text Extraction App")

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    temp_image_path = "uploaded_image.jpeg"
    image.save(temp_image_path)

    @st.cache_data
    def create_rag_index(image_path):
        RAG.index(
            input_path=image_path,
            index_name="image_index",
            store_collection_with_index=True,
            overwrite=True,
        )

    create_rag_index(temp_image_path)

    text_query = st.text_input("Enter your text query")

    if st.button("Search and Extract Text"):
        if text_query:
            results = RAG.search(text_query, k=1, return_base64_results=True)

            image_data = base64.b64decode(results[0].base64)
            image = Image.open(BytesIO(image_data))
            st.image(image, caption="Result Image", use_column_width=True)

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": "extract text"}
                    ]
                }
            ]

            text_prompt = processor.apply_chat_template(messages, add_generation_prompt=True)

            inputs = processor(
                text=[text_prompt],
                images=[image],
                padding=True,
                return_tensors="pt"
            )

            inputs = inputs.to(model.device)

            with torch.no_grad():
                output_ids = model.generate(**inputs, max_new_tokens=1024)

            generated_ids = output_ids[:, inputs.input_ids.shape[1]:]

            output_text = processor.batch_decode(
                generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
            )[0]

            # Highlight the queried text
            def highlight_text(text, query):
                highlighted_text = text
                for word in query.split():
                    pattern = re.compile(re.escape(word), re.IGNORECASE)
                    highlighted_text = pattern.sub(lambda m: f'<span style="background-color: yellow;">{m.group()}</span>', highlighted_text)
                return highlighted_text

            highlighted_output = highlight_text(output_text, text_query)

            st.subheader("Extracted Text (with query highlighted):")
            st.markdown(highlighted_output, unsafe_allow_html=True)
        else:
            st.warning("Please enter a query.")
else:
    st.info("Upload an image to get started.")