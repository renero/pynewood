import ipywidgets as widgets

from IPython.display import display, Markdown
from typing import List, Dict


def multiple_select_items(options: Dict[str, bool]):
    """
    Displays checkboxes for those items in the `options` arg. If the value of the
    keys are true, then they appear marked as default.

    Example
        >>> options = {'sachs':True, 'sachs_long':False, 'toy':False, 'insurance':False}
        >>> checkboxes = multiple_select_items(options)
        ...
        >>> to_reprocess = items_selected(checkboxes)

    Args:
        options (dict(str, bool)): dict of options and whether they should appear
            marked or not.

    Returns:
        widgets.Checkbox

    """
    display(Markdown('What items to select?'))
    checkboxes = [
        widgets.Checkbox(value=options[label], description=label) for label in options
    ]
    output = widgets.VBox(children=checkboxes)
    display(output)
    return checkboxes


def items_selected(selection: widgets.Checkbox):
    """ Returns what values have been selected from checkboxes list (selection) """
    selected_data = []
    for i in range(0, len(selection)):
        if selection[i].value == True:
            selected_data = selected_data + [selection[i].description]

    return selected_data
