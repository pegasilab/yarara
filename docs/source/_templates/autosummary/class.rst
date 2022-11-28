{{ name | escape | underline}}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}
   :show-inheritance:
   :members:
   :undoc-members:
   :inherited-members:
   :special-members: __call__, __add__, __mul__, __init__

   {% block attributes %}
   {% if attributes %}
   .. rubric:: {{ _('Attributes') }}

   *Note: attributes inherited from parent classes are not shown here, if any*

   .. autosummary::
   {% for item in attributes %}
      ~{{ name }}.{{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}

   {% block methods %}
   {% if methods %}
   .. rubric:: {{ _('Methods') }}

   .. autosummary::
      :nosignatures:
   {% for item in methods %}
      {%- if not item.startswith('_') %}
      ~{{ name }}.{{ item }}
      {%- endif -%}
   {%- endfor %}
   {% endif %}
   {% endblock %}


   .. rubric:: List of members of {{ objname }}
