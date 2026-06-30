from docutils import nodes


def _inline_rendered_return_type_fields(app, doctree, docname):
    """
    Rewrite Sphinx field lists so return types appear inline with return descriptions.

    Why this hook exists:
    - With ``autodoc_typehints = "both"``, Sphinx renders two separate fields for return information:
      1) ``RETURNS:`` (the textual description)
      2) ``RETURN TYPE:`` (the annotation/type).
      This visually takes up a lot of space.
    - We instead want the compact format: ``RETURNS: <type> - <description>``.

    What this hook does:
    - Runs at ``doctree-resolved`` after autodoc and napoleon have built the final docutils tree for a page.
    - For each field-list block, if both ``Returns`` and ``Return type`` are present, it moves the rendered return-type
      nodes to the start of the ``Returns`` body, adds a separator, and removes the standalone ``Return type`` field.
    - We copy rendered nodes (not plain text) so type cross-references continue to work, e.g. links to ``Tensor`` docs
      remain clickable.
    """

    def _get_field_name_and_body(field):
        # A docutils field consists of:
        # - field_name: label such as "Returns" or "Return type"
        # - field_body: the content associated with that label
        field_name = None
        field_body = None
        for child in field.children:
            if isinstance(child, nodes.field_name):
                field_name = child
            elif isinstance(child, nodes.field_body):
                field_body = child
        return field_name, field_body

    # Sphinx renders Args/Returns-style sections as docutils field_list blocks.
    for field_list in doctree.findall(nodes.field_list):
        returns_field = None
        return_type_field = None

        # Find sibling fields named "Returns" and "Return type" in this field list.
        for field in [child for child in field_list.children if isinstance(child, nodes.field)]:
            field_name, _ = _get_field_name_and_body(field)
            if field_name is None:
                continue

            normalized_name = field_name.astext().strip().lower()
            if normalized_name == "returns":
                returns_field = field
            elif normalized_name == "return type":
                return_type_field = field

        if returns_field is None or return_type_field is None:
            # This list has nothing to rewrite.
            continue

        _, returns_body = _get_field_name_and_body(returns_field)
        _, return_type_body = _get_field_name_and_body(return_type_field)
        if returns_body is None or return_type_body is None:
            continue

        # Return type content is usually wrapped in a paragraph node.
        return_type_para = next(
            (child for child in return_type_body.children if isinstance(child, nodes.paragraph)),
            None,
        )
        if return_type_para is not None:
            return_type_nodes = [node.deepcopy() for node in return_type_para.children]
        else:
            return_type_nodes = [node.deepcopy() for node in return_type_body.children]

        if not return_type_nodes:
            # Empty Return type field: remove it and continue.
            field_list.remove(return_type_field)
            continue

        # We prepend return type nodes to the first paragraph in Returns.
        first_paragraph = next(
            (child for child in returns_body.children if isinstance(child, nodes.paragraph)),
            None,
        )
        if first_paragraph is None:
            first_paragraph = nodes.paragraph()
            returns_body.insert(0, first_paragraph)

        # Rebuild as: <type nodes> – <existing description nodes>
        description_nodes = list(first_paragraph.children)
        first_paragraph.clear()
        first_paragraph.extend(return_type_nodes)
        if description_nodes:
            first_paragraph += nodes.Text(" \u2013 ")
            first_paragraph.extend(description_nodes)

        # Remove the standalone field because its content is now inlined.
        field_list.remove(return_type_field)
