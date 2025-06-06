import importlib.util, random, time
from isolation import Board, Game, MCTSPlayer, RandomPlayer, Colour

def run_match(red_agent, blue_agent, board_size, red_starts=True):
    """
    Play one game. If red_starts=False, swap the agents so blue goes first.
    Returns (winner, moves).
    """
    board = Board(*board_size)
    if red_starts:
        R = red_agent; B = blue_agent; current = Colour.RED
    else:
        R = blue_agent; B = red_agent; current = Colour.RED  # but R/B roles reversed
        # at the end, if R wins, that means “blue_agent” originally won, etc.

    moves = 0
    while True:
        legal = board.moves_for(current)
        if not legal:
            winner = current.flip()
            break

        if current == Colour.RED:
            move = R.choose_action(board.duplicate(), current)
            B.register_opponent_action(move)
        else:
            move = B.choose_action(board.duplicate(), current)
            R.register_opponent_action(move)

        board.apply_move(current, move)
        current = current.flip()
        moves += 1

    # convert winner back to global perspective
    if not red_starts:
        winner = Colour.RED if winner == Colour.BLUE else Colour.BLUE
    return winner, moves

def baseline_mcts_vs_random(board_size, time_values, c_values, num_games=10):
    print(f"\n── Baseline: MCTS vs Random on board {board_size} ──")
    # experiments for different time values at fixed c
    c0 = 0.6
    for t in time_values:
        red_wins = 0; total_moves = 0
        for i in range(num_games):
            # alternate who starts:
            red_starts = (i < num_games // 2) # first half red starts, second half blue starts
            red = MCTSPlayer(time_limit=t, c_coefficient=c0)
            blue = RandomPlayer()
            w, m = run_match(red, blue, board_size, red_starts=red_starts)
            if w == Colour.RED: red_wins += 1
            total_moves += m
        print(f" time={t:>4.4f}s  c={c0:>4.2f}  "
              f"Win%={red_wins/num_games:>5.1%}  avg_len={total_moves/num_games:>4.1f}")


    # experiments for different c values at fixed t
    t0 = 0.1
    for c in c_values:
        red_wins = 0; total_moves = 0
        for i in range(num_games):
            red_starts = (i < num_games // 2)
            red = MCTSPlayer(time_limit=t0, c_coefficient=c)
            blue = RandomPlayer()
            w, m = run_match(red, blue, board_size, red_starts=red_starts)
            if w == Colour.RED: red_wins += 1
            total_moves += m
        print(f" time={t0:>4.2f}s  c={c:>4.2f}  "
              f"Win%={red_wins/num_games:>5.1%}  avg_len={total_moves/num_games:>4.1f}")

def mcts_vs_mcts_tests(board_size, time_pairs, c_pairs, num_games=10):
    print(f"\n── MCTS vs MCTS on board {board_size} ──")
    # experiments for different time values at fixed c
    c0 = 0.6
    for (t_red, t_blue) in time_pairs:
        red_wins = 0; total_moves = 0
        for i in range(num_games):
            red = MCTSPlayer(time_limit=t_red, c_coefficient=c0)
            blue= MCTSPlayer(time_limit=t_blue,c_coefficient=c0)
            red_starts = (i < num_games//2)
            w, m = run_match(red, blue, board_size, red_starts=red_starts)
            if w == Colour.RED: red_wins += 1
            total_moves += m
        print(f" TRed={t_red:>4.2f}s  TBlue={t_blue:>4.2f}s  c={c0:>4.2f}  "
              f"Red‐win%={red_wins/num_games:>5.1%}  avg_len={total_moves/num_games:>4.1f}")

    # # experiments for different c values at fixed t
    # t0 = 0.10
    # for (c_red, c_blue) in c_pairs:
    #     red_wins = 0; total_moves = 0
    #     for i in range(num_games):
    #         red = MCTSPlayer(time_limit=t0, c_coefficient=c_red)
    #         blue= MCTSPlayer(time_limit=t0, c_coefficient=c_blue)
    #         red_starts = (i < num_games//2)
    #         w, m = run_match(red, blue, board_size, red_starts=red_starts)
    #         if w == Colour.RED: red_wins += 1
    #         total_moves += m
    #     print(f" time={t0:>4.2f}s  cRed={c_red:>4.2f}  cBlue={c_blue:>4.2f}  "
    #           f"Red‐win%={red_wins/num_games:>5.1%}  avg_len={total_moves/num_games:>4.1f}")


if __name__ == "__main__":
    board_sizes = [(5,5), (7, 5), (8, 8)]

    # for MCTS vs Random
    time_values = [0.0001, 0.001, 0.01, 0.1, 0.2]
    c_values = [0.0, 0.2, 0.6, 1.0, 1.5, 2.0, 4.0]

    # For MCTS vs MCTS:
    time_pairs = [(0.0001, 0.001), (0.0001, 0.1), (0.1, 0.2), (0.1, 0.4)]
    time_pairs = [(0.0001, 0.1)]
    c_pairs    = [(0.4, 1.0), (0.6, 1.0), (0.6, 1.4), (0.6, 2.0), (0.6, 4.0)]

    num_games = 100

    for bs in board_sizes:
        # baseline_mcts_vs_random(bs, time_values, c_values, num_games=num_games)
        mcts_vs_mcts_tests(bs, time_pairs, c_pairs, num_games=num_games)

    print("\n>>> All tests complete! <<<")