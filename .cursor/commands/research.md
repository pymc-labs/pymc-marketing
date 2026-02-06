# Research over a request

Structure a research based on the user request. Identify what must change in order to complete the task.

You must:
- Create a description with the user request, being explicit without overview, citing textually.
- Check python files related to the request.
- Identify what should be change.
- Check what possible other files could be affected.
- Make reasonable technical decisions based on research
- If multiple approaches exist, specify each possible option, extend the research to explain their pros and cons, then choose the most practical one.
- Document assumptions in the plan.

**No user interaction required:**
    - DO NOT ask for clarifications or wait for input

## Python test validations
Create temporal python files, check the logics you are thinking on the root folder. Execute this python code, and based on the output decide what should be done.

Once you test the code out of the code base using this temporal python files and functions, document the approach using natural language.

**Use the python files with objective:**
- Validate your assumption about how the code works.
- Compare code compilation (we must pick code which is faster).
- Check code behaviour inputs and outputs.

Delete the files once, you finish your testing.

## Document base
Your goal is create a folder under `.cursor/plans`. You must save all your discoveries under `.cursor/plans/{plan_name_folder}/research` this must be a file `.md`.

You only need to provide the research plan and a possible tested solution.

No other `md` files are needed, everything must be collapse in the research file.
