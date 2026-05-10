{# Members that should not be enumerated in either Methods or Attributes:
   - str.maketrans / translate / format / format_map etc. leak in via StrEnum
     subclasses (ConvMode, WeibullType, CovFunc, PeriodicCovFunc); autodoc
     cannot format their overloaded C-level signatures.
   - rv_op on PyMC Distribution subclasses is a classmethod descriptor;
     autodoc raises 'list assignment index out of range' on its signature
     and 'failed to import object' on its attribute reference. #}
{% set excluded_members = [
    "maketrans", "translate", "format", "format_map",
    "encode", "decode", "removeprefix", "removesuffix",
    "rv_op",
] %}
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
   {% if attributes %}
   .. rubric:: Attributes

   .. autosummary::
   {% for item in attributes %}
   {%- if item not in excluded_members %}
      ~{{ name }}.{{ item }}
   {%- endif %}
   {%- endfor %}
   {% endif %}
   {% endblock %}
