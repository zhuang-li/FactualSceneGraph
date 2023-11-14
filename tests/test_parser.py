from factual_scene_graph.parser.scene_graph_parser import SceneGraphParser
from factual_scene_graph.utils import space_out_symbols_in_graph, tprint, clean_graph_string


def test_clean_graph_string():
    # Test normal case
    assert clean_graph_string("( bench , is , woo") == "( bench )"
    # Test with extra spaces
    assert clean_graph_string("( bench , is , wooden ) ,") == "( bench , is , wooden )"
    # Test with multiple relations
    assert clean_graph_string("( bench , is , wooden ) , ( bench , v:faces , ") == "( bench , is , wooden )"
    # Test empty string
    assert clean_graph_string("") == ""
    print("All tests for clean_graph_string passed!")

def test_space_out_symbols_in_graph():
    # Test normal case
    assert space_out_symbols_in_graph("(bench,is,wooden)") == "( bench , is , wooden )"
    # Test with extra spaces
    assert space_out_symbols_in_graph(" (bench ,is,wooden) ") == "( bench , is , wooden )"
    # Test with multiple relations
    assert space_out_symbols_in_graph("(bench,is,wooden),(bench,v:faces,sea)") == "( bench , is , wooden ) , ( bench , v:faces , sea )"
    # Test empty string
    assert space_out_symbols_in_graph("") == ""
    print("All tests for space_out_symbols_in_graph passed!")

def test_scene_graph_parser():
    # Assuming SceneGraphParser is correctly instantiated as `parser`
    # Test normal input
    parser = SceneGraphParser('lizhuang144/flan-t5-base-VG-factual-sg', device='cuda:0')
    text_graph = parser.parse(["2 beautiful pigs are flying on the sky with 2 bags on their backs"],return_text=True)

    print(text_graph[0])
    text_graph = parser.parse(["2 beautiful and strong pigs are flying on the sky with 2 bags on their backs",
                              "a blue sky"], max_output_len=128, return_text=True)
    print(text_graph[0])

    graph_obj = parser.parse(["2 beautiful and strong pigs are flying on the sky with 2 bags on their backs",
                              "a blue sky"], max_output_len=128, return_text=False)
    print(graph_obj[0])

    tprint(graph_obj[0])

    graph_obj = parser.parse(["boy"], max_output_len=16, return_text=False)
    print(graph_obj[0])

    tprint(graph_obj[0])


if __name__ == "__main__":
    test_space_out_symbols_in_graph()
    test_clean_graph_string()
    test_scene_graph_parser()