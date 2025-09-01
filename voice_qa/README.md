# Repo Details
 - has two project in a single repo
 - none of the code,scripts in this repo is AI generated except for the debugging helps.
 - read both the README.md in the root dir and the voice_qa dir
 - 
# Voice-Driven Q&A Application Using LangChain
- Accepts 3+ web URLs, scrapes combined content.
- Voice input questions via microphone.
- Textual and spoken audio answers.
- Gracefully handles inaccessible URLs without crashing.

# Agent driven LLM 
 - this agent when asked anny question after providing a number of website 
can assess the websites and use tools that can also assess more websites 
autonomously
 - the requirement for this script is given in the requirements.txt in the root file
 - run python main.py in the root directory for running the agent
 - this agent works in the cli 
 - agent llm is using Ollama for my convenience and situation

# Ollama Setup 
 - install ollama installer through the official site **https://ollama.com/**
 - then the pull the model I used which is _mistral_ -> ollama pull mistral 
 - the mistral is a 7B parameter model so if you don't have the hardware setup for this 
config please do chose your own model like llama 3B or similar that has lower parameter sizze
 - so now for the embedding model i have used _nomic-embed-text_ -> ollama pull nomic-embed-text


# Environment Setting
 - I have used _uv_ for maintaining the environment for the specific project
 - so it is now according to the user to configure his own for their convinience
 - I have included the requirements.txt in the root install it using either
 - + -> pip install -r requirements.txt or conda install -r requirements.txt or uv install -r requirements.txt 



---

 If you need any help, reach out:  
- [Email](mailto:fauzan41527@gmail.com)  
- [WhatsApp](https://wa.me/+919567117031)  

  
