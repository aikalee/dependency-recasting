def structured_tokens_to_linearzied(items):
    output = []
    for item in items:
        output.extend(item.get("left", []))
        output.append(item["token"])
        output.extend(item.get("right", []))
    return " ".join(output)
    
    