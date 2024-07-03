:: mcp & ac
python main.py --k_shot 0 --mcp
python main.py --k_shot 0 --mcp --torler
python main.py --k_shot 1 --mcp
python main.py --k_shot 1 --mcp --torler
python main.py --k_shot 5 --mcp
python main.py --k_shot 5 --mcp --torler

:: cp & ac
python main.py --k_shot 0 --cp
python main.py --k_shot 1 --cp
python main.py --k_shot 5 --cp

:: mcp & ae
python main.py --dataset ae --k_shot 0 --mcp
python main.py --dataset ae --k_shot 0 --mcp --torler
python main.py --dataset ae --k_shot 1 --mcp
python main.py --dataset ae --k_shot 1 --mcp --torler
python main.py --dataset ae --k_shot 5 --mcp
python main.py --dataset ae --k_shot 5 --mcp --torler

:: cp & ae
python main.py --dataset ae --k_shot 0 --cp
python main.py --dataset ae --k_shot 1 --cp
python main.py --dataset ae --k_shot 5 --cp
