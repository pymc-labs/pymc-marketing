{{ name | escape | underline}}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}

   {% block methods %}
   {% if methods %}

   .. rubric:: Methods

   .. autosummary::
      :toctree: classmethods

   {# Skip members that are inherited-but-noisy or fail autodoc introspection.
      - str.maketrans / translate / format / format_map etc. leak in via
        StrEnum subclasses (ConvMode, WeibullType, CovFunc, PeriodicCovFunc).
        autodoc cannot format their overloaded C-level signatures.
      - rv_op on PyMC Distribution subclasses is a classmethod with a dynamic
        signature; autodoc raises 'list assignment index out of range'. #}
   {% set excluded_members = [
       "maketrans", "translate", "format", "format_map",
       "encode", "decode", "removeprefix", "removesuffix",
       "rv_op",
   ] %}
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
   {# Same exclusions as the methods block: rv_op shows up as an attribute on
      Distribution subclasses (it's a classmethod descriptor) and autodoc fails
      to import it, producing 'failed to import object ...rv_op' warnings. #}
   {% for item in attributes %}
   {%- if item not in excluded_members %}
      ~{{ name }}.{{ item }}
   {%- endif %}
   {%- endfor %}
   {% endif %}
   {% endblock %}
