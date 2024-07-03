@REM :: mcp & ac
@REM python main.py --k_shots 0 --mcp
@REM python main.py --k_shots 0 --mcp --torler
@REM python main.py --k_shots 1 --mcp
@REM python main.py --k_shots 1 --mcp --torler
@REM python main.py --k_shots 5 --mcp
@REM python main.py --k_shots 5 --mcp --torler

@REM :: cp & ac
@REM python main.py --k_shots 0 --cp
@REM python main.py --k_shots 1 --cp
@REM python main.py --k_shots 5 --cp

:: mcp & ae
python main.py --dataset ae --k_shots 0 --mcp
python main.py --dataset ae --k_shots 0 --mcp --torler
python main.py --dataset ae --k_shots 1 --mcp
python main.py --dataset ae --k_shots 1 --mcp --torler
python main.py --dataset ae --k_shots 5 --mcp
python main.py --dataset ae --k_shots 5 --mcp --torler

:: cp & ae
python main.py --dataset ae --k_shots 0 --cp
python main.py --dataset ae --k_shots 1 --cp
python main.py --dataset ae --k_shots 5 --cp
