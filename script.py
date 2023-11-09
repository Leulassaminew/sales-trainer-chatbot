!git clone https://github.com/taprosoft/llm_finetuning/
!cd llm_finetuning
!pip install -r requirements.txt
!mkdir models
!python download_model.py TheBloke/Llama-2-13B-chat-GPTQ --branch gptq-4bit-128g-actorder_True	
cd ..
!git clone https://github.com/Leulassaminew/sales-trainer-chatbot.git
!cd sales-trainer-chatbot
!pip install -r requirements.txt
!pip install gradio > /dev/null
