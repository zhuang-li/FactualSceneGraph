def get_seg_list(graphs):
    if isinstance(graphs, str):
        seg_list = [scene_seg.replace('(', '').replace(')', '').strip() for scene_seg in format_scene_graph(graphs).split(') , (')]
    elif isinstance(graphs, list):
        seg_list = []
        for graph in graphs:
            seg_list.extend([scene_seg.replace('(', '').replace(')', '').strip() for scene_seg in format_scene_graph(graph).split(') , (')])
    else:
        raise ValueError('input should be either a string or a list of strings')
    return list(set(seg_list))

def format_scene_graph(graph_str):
    return " ".join([item for item in graph_str.replace('(', ' ( ').replace(')', ' ) ').replace(',', ' , ').split() if item != ''])