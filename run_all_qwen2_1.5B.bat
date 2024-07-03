:: mcp & ac
python main.py --model qwen2-1.5B-ins --k_shots 0 --mcp
python main.py --model qwen2-1.5B-ins --k_shots 0 --mcp --torler
python main.py --model qwen2-1.5B-ins --k_shots 1 --mcp
python main.py --model qwen2-1.5B-ins --k_shots 1 --mcp --torler
python main.py --model qwen2-1.5B-ins --k_shots 5 --mcp
python main.py --model qwen2-1.5B-ins --k_shots 5 --mcp --torler

:: cp & ac
python main.py --model qwen2-1.5B-ins --k_shots 0 --cp
python main.py --model qwen2-1.5B-ins --k_shots 1 --cp
python main.py --model qwen2-1.5B-ins --k_shots 5 --cp

:: mcp & ae
python main.py --model qwen2-1.5B-ins --dataset ae --k_shots 0 --mcp
python main.py --model qwen2-1.5B-ins --dataset ae --k_shots 0 --mcp --torler
python main.py --model qwen2-1.5B-ins --dataset ae --k_shots 1 --mcp
python main.py --model qwen2-1.5B-ins --dataset ae --k_shots 1 --mcp --torler
python main.py --model qwen2-1.5B-ins --dataset ae --k_shots 5 --mcp
python main.py --model qwen2-1.5B-ins --dataset ae --k_shots 5 --mcp --torler

:: cp & ae
python main.py --model qwen2-1.5B-ins --dataset ae --k_shots 0 --cp
python main.py --model qwen2-1.5B-ins --dataset ae --k_shots 1 --cp
python main.py --model qwen2-1.5B-ins --dataset ae --k_shots 5 --cp
