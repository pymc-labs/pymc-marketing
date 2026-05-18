{# Members that should not be enumerated in either Methods or Attributes
   because they are inherited from third-party base classes whose docstrings
   trigger noisy docutils/numpydoc warnings, or because autodoc cannot
   introspect their signatures.

   - str.* methods (maketrans / translate / format / format_map etc.) leak
     in via StrEnum subclasses (ConvMode, WeibullType, CovFunc,
     PeriodicCovFunc); autodoc cannot format their overloaded C-level
     signatures.
   - rv_op on PyMC Distribution subclasses is a classmethod descriptor;
     autodoc raises 'list assignment index out of range' on its signature
     and 'failed to import object' on its attribute reference.
   - pydantic.BaseModel.* methods leak in via the many Field-based config
     classes (HSGPBase, BassPriors, etc.); their pydantic-side docstrings
     have malformed RST (Definition list / Block quote without trailing
     blank line) which inflates our docs warnings ~16x via inheritance.
   - mlflow.pyfunc.model.PythonModel.* methods leak in via PyFuncModel
     wrappers in mlflow.py; same problem as pydantic. #}
{% set excluded_members = [
    "maketrans", "translate", "format", "format_map",
    "encode", "decode", "removeprefix", "removesuffix",
    "rv_op",
    "model_construct", "model_copy", "model_dump", "model_dump_json",
    "model_validate", "model_validate_json", "model_validate_strings",
    "model_json_schema", "model_post_init", "model_rebuild",
    "predict_stream", "load_context",
    "construct", "copy", "dict", "from_orm", "json",
    "parse_obj", "parse_raw", "parse_file",
    "schema", "schema_json", "update_forward_refs", "validate",
    "model_parametrized_name",
] %}
{# Pydantic models have their fields documented inline by autopydantic_model
   (see the source-read hook in conf.py), so the Attributes summary table is
   skipped to avoid duplicating user fields and listing pydantic internals
   (model_config, model_fields, model_extra, ...). Detect pydantic by the
   presence of `model_fields` in the attributes list, which is reliable in
   pydantic v2 and never present on non-pydantic classes. #}
{% set is_pydantic_model = "model_fields" in attributes %}
{{ name | escape | underline}}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}

   {% block methods %}
   {% if methods %}

   .. rubric:: Methods

   .. autosummary::
      :toctree: classmethods

   {% for item in methods %}
   {%- if item not in excluded_members %}
      {{ objname }}.{{ item }}
   {%- endif %}
   {%- endfor %}
   {% endif %}
   {% endblock %}

   {% block attributes %}
   {% if attributes and not is_pydantic_model %}
   .. rubric:: Attributes

   .. autosummary::
   {% for item in attributes %}
   {%- if item not in excluded_members %}
      ~{{ name }}.{{ item }}
   {%- endif %}
   {%- endfor %}
   {% endif %}
   {% endblock %}
