---
inclusion: always
---
<!------------------------------------------------------------------------------------
   Add rules to this file or a short description and have Kiro refine them for you.
   
   Learn about inclusion modes: https://kiro.dev/docs/steering/#inclusion-modes
------------------------------------------------------------------------------------->

All code MUST BE written in English (variables, classes, attributes, etc.);

All documentation in the form of comments or docstrings MUST BE in Brazilian Portuguese;

WHEN coding based on requirements and/or design files, the requirement ids MUST NOT be in comments, for example:

```python
# --- "Data a programar" (Req 1.5, 1.6, 3.1, 4.1, 7.1, 7.2) ---
# A janela de referência é a recebida em runtime, não o expediente do turno (Req 7.1).
   window_start = pd.to_datetime(data_dict["__start_prog__"]).date()
   window_end = pd.to_datetime(data_dict["__end_prog__"]).date()
```

All created methods must include type hints and docstrings following the Google Style Guide. The type hints must also be included in the docstring.
