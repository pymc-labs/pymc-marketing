{# Classes that should not get their own autosummary page. Deprecated
   backward-compat aliases (e.g. OptimizerCompatibleModelWrapper, an alias of
   OptimizerCompatibleModel) resolve to the same class object under a second
   name; autodoc then appends an "alias of ..." trailer with no blank line,
   which docutils flags as "Explicit markup ends without a blank line"
   (issue #2660). The canonical class still gets its page. #}
{% set excluded_classes = [
    "OptimizerCompatibleModelWrapper",
] %}
{{ name | escape | underline}}

.. automodule:: {{ fullname }}

   {% block attributes %}
   {% if attributes %}
   .. rubric:: {{ _('Module Attributes') }}

   .. autosummary::
   {% for item in attributes %}
      {{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}

   {% block functions %}
   {% if functions %}
   .. rubric:: {{ _('Functions') }}

   .. autosummary::
      :toctree:

   {% for item in functions %}
      {{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}

   {% block classes %}
   {% if classes %}
   .. rubric:: {{ _('Classes') }}

   .. autosummary::
      :toctree:

   {% for item in classes %}
      {% if not item.endswith("RV") and item not in excluded_classes %}
      {{ item }}
      {% endif %}
   {%- endfor %}
   {% endif %}
   {% endblock %}

   {% block exceptions %}
   {% if exceptions %}
   .. rubric:: {{ _('Exceptions') }}

   .. autosummary::
   {% for item in exceptions %}
      {{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}

{% block modules %}
{% if modules %}
.. rubric:: Modules

.. autosummary::
   :toctree:
   :recursive:
{% for item in modules %}
   {{ item }}
{%- endfor %}
{% endif %}
{% endblock %}
