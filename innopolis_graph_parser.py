from bs4 import BeautifulSoup
from datetime import datetime

def main():
    f = open('data/map-2.osm', 'r')

    soup = BeautifulSoup("".join(f.readlines()), features="xml")

    nodes = {}

    for node in soup.find_all('node'):
        nodes[node["id"]] = (node["lat"], node["lon"])

    ways = []

    for way in soup.find_all('way'):
        time = datetime.strptime(way["timestamp"], "%Y-%m-%dT%H:%M:%SZ")
        arr = way.find_all('nd')

        flag = True
        for i in way.find_all('tag'):
            if i["k"] == "highway":
                flag = False
        
        if flag:
            continue

        for i in range(len(arr) - 1):
            ways.append((time, arr[i]["ref"], arr[i + 1]["ref"]))
    
    ways.sort()

    f = open('data/dubki.graph', 'w')

    f.write(f"Nodes\t{len(nodes)}\n")
    for i in nodes:
        f.write(f"{i}\t{nodes[i][0]}\t{nodes[i][1]}\n")
    
    f.write(f"Edges\t{len(ways)}\n")
    for i in ways:
        f.write(f"{i[0].date()}\t{i[1]}\t{i[2]}\n")


if __name__ == '__main__':
    main()
