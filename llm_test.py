import modal 

app = modal.App.lookup('llm-runner')
run_llm = app._add_function("run_llm")

response = run_llm.remote('tell me about AI') 
print(response)