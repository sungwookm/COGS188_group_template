def getMoveDict():
    codes, i = {}, 0
    for nSqr in range(1,8):
        for dir in ["N","NE","E","SE","S","SW","W","NW"]:
            codes[(nSqr,dir)] = i
            i += 1

    for two in ["N", "S"]:
        for one in ["E", "W"]:
            codes[("Knight", two, one)], i = i, i+1
    
    for two in ["E", "W"]:
        for one in ["N", "S"]:
            codes[("Knight", two, one)], i = i, i+1

    for move in ["N","NW", "NE"]:
        for promo in ["Rook", "Knight", "Bishop"]:
            codes[("under_promo", move, promo)], i = i, i+1

    assert len(codes) == 73, "Error: Incorrect number of move encodings"
    print(codes)
    return codes

getMoveDict()