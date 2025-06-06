from abc import abstractmethod
from enum import Enum
import itertools
import random
import time
from typing import Optional, Protocol
import math


class Colour(Enum):
    RED = 'R'
    BLUE = 'B'

    def flip(self) -> 'Colour':
        return Colour.RED if self == Colour.BLUE else Colour.BLUE


class Board:
    def __init__(self, width: int, height: int):
        self.width: int = width
        self.height: int = height
        self.positions: dict[tuple[int, int], str] = dict()
        self.red_position: Optional[tuple[int, int]] = None
        self.blue_position: Optional[tuple[int, int]] = None
        self._prepare_board()

    def _prepare_board(self):
        for i in range(self.width):
            for j in range(self.height):
                self.positions[(i, j)] = '.'

    def __str__(self):
        representation = '\\ ' + ' '.join([str(i + 1) for i in range(self.width)]) + '\n'
        for j in range(self.height):
            representation += (chr(ord('A') + j) + ' ' + ' '.join([self.positions[i, j] for i in range(self.width)]))
            if j < self.height - 1:
                representation += '\n'
        return representation

    def moves_for(self, current_player: Colour) -> list[tuple[int, int]]:
        result = []
        player_position = self._player_position(current_player)
        if player_position is None:
            for position in self.positions:
                if self.positions[position] == '.':
                    result.append(position)
        else:
            directions = list(itertools.product([-1, 0, 1], repeat=2))
            directions.remove((0, 0))
            for dx, dy in directions:
                px, py = player_position
                px, py = px + dx, py + dy
                while 0 <= px < self.width and 0 <= py < self.height:
                    potential_position = px, py
                    if self.positions[potential_position] == '.':
                        result.append(potential_position)
                        px, py = px + dx, py + dy
                    else:
                        break
        return result

    def apply_move(self, current_player: Colour, move: tuple[int, int]) -> None:
        player_position = self._player_position(current_player)
        if player_position is not None:
            self.positions[player_position] = '#'
        self.positions[move] = current_player.value
        self._update_player_position(current_player, move)

    def _player_position(self, current_player: Colour) -> tuple[int, int]:
        return self.red_position if current_player == Colour.RED else self.blue_position

    def _update_player_position(self, current_player: Colour, new_position: tuple[int, int]) -> None:
        if current_player == Colour.RED:
            self.red_position = new_position
        else:
            self.blue_position = new_position

    def to_state_str(self) -> str:
        positions_in_order = []
        for j in range(self.height):
            for i in range(self.width):
                positions_in_order.append(self.positions[(i, j)])
        return f"{self.width}_{self.height}_{''.join(positions_in_order)}"

    @staticmethod
    def from_state_str(state_str: str) -> 'Board':
        width, height, positions = state_str.split('_')
        width, height = int(width), int(height)
        board = Board(width, height)
        for j in range(height):
            for i in range(width):
                position = positions[j * width + i]
                board.positions[(i, j)] = position
                if position == Colour.RED.value:
                    board.red_position = (i, j)
                elif position == Colour.BLUE.value:
                    board.blue_position = (i, j)
        return board

    # handy for exploring “what‐if” moves without mutating the original Board
    def duplicate(self) -> 'Board':
        return self.from_state_str(self.to_state_str())


class Player(Protocol):
    @abstractmethod
    def choose_action(self, board: Board, current_player: Colour) -> tuple[int, int]:
        raise NotImplementedError

    @abstractmethod
    def register_opponent_action(self, action: tuple[int, int]) -> None:
        raise NotImplementedError


class Game:
    # zasady tego wariantu gry w izolację, są bardzo proste
    # zasady:
    #  * jest dwóch graczy, czerwony i niebieski, czerwony porusza się pierwszy
    #  * każdy gracz ma dokładnie jeden pionek w swoim kolorze ('R' lub 'B')
    #  * plansza jest prostokątem, w swoim pierwszym ruchu każdy gracz może położyć pionek na jej dowolnym pustym polu
    #  * w kolejnych ruchach gracze naprzemiennie przesuwają swoje pionki
    #     * pionki poruszają się jak hetmany szachowe (dowolna liczba pól w poziomie, pionie, lub po skosie)
    #     * pole, z którego pionek startował jest usuwane z planszy ('.' zastępuje '#') i trwale zablokowane
    #     * zarówno pionek innego gracza jak i zablokowane pola uniemożliwiają dalszy ruch (nie da się ich przeskoczyć)
    #  * jeżeli gracz musi wykonać ruch pionkiem, a nie jest to możliwe (każdy z ośmiu kierunków zablokowany)...
    #  * ...to taki gracz przegrywa (a jego przeciwnik wygrywa ;])
    def __init__(self, red: Player, blue: Player, board: Board, current_player: Colour = Colour.RED):
        self.red: Player = red
        self.blue: Player = blue
        self.board: Board = board
        self.current_player: Colour = current_player
        self.finished: bool = False
        self.winner: Optional[Colour] = None

    def run(self, verbose=False):
        if verbose:
            print()
            print(self.board)

        while not self.finished:
            legal_moves = self.board.moves_for(self.current_player)
            if len(legal_moves) == 0:
                self.finished = True
                self.winner = Colour.BLUE if self.current_player == Colour.RED else Colour.RED
                break

            player = self.red if self.current_player == Colour.RED else self.blue
            opponent = self.red if self.current_player == Colour.BLUE else self.blue
            move = player.choose_action(self.board, self.current_player)
            opponent.register_opponent_action(move)
            self.board.apply_move(self.current_player, move)
            self.current_player = self.current_player.flip()

            if verbose:
                print()
                print(self.board)

        if verbose:
            print()
            print(f"WINNER: {self.winner.value}")


class RandomPlayer(Player):
    """
    # A trivial agent that picks uniformly at random among all legal moves.
    """
    def choose_action(self, board: Board, current_player: Colour) -> tuple[int, int]:
        legal_moves = board.moves_for(current_player)
        return random.sample(legal_moves, 1)[0]

    def register_opponent_action(self, action: tuple[int, int]) -> None:
        pass


class MCTSNode: # this represents the current state of the game
    def __init__(self, board: Board, current_player: Colour, c_coefficient: float):
        self.parent: Optional[MCTSNode] = None
        self.leaf: bool = True
        self.terminal: bool = False # node where no moves remain
        self.times_chosen: int = 0
        self.value: float = 0.5 # how “good” this node is for the player who just moved
        self.children: dict[tuple[int, int], MCTSNode] = dict() # dict keys (MCTSNode(s)) are the available actions in this state
        self.board: Board = board
        self.current_player: Colour = current_player
        self.c_coefficient: float = c_coefficient # in UCB defines the strength of exploration

    def select(self, final=False) -> tuple[int, int]: # final here is a flag for whether the planning stage is finished
        # tutaj należy wybrać (i zwrócić) najlepszą możliwą akcję (w oparciu o aktualną wiedzę)
        # podpowiedzi:
        #  * klucze w słowniku `self.children` to pula dostępnych akcji
        #  * każdą z nich należy ocenić zgodnie z techniką UCB (tak jakby był to problem wielorękiego bandyty)
        #  * ocena akcji zależy od:
        #     * jej wartościowania (`self.value`)
        #     * oraz tego jak często była wybierana (`self.times_chosen`) w porównaniu z rodzicem
        #     (a to dlatego, że z rodzica startujemy symulację, więc siłą rzeczy on najczęściej "chosen"
        #     (bo skoro w nim jesteśmy to musiał być najczęściej wybierany?) no i jak dana możliwa akcja była rzadziej wybierana,
        #     to będzie miała wyższy współczynnik przemnażany przez c a tego dpokładnie chcemy, bo chcemy zwi ększyć jej eksplorację
        #     * odpowiednie wartości przechowują węzły-dzieci przyporządkowane w słowniku kluczom-akcjom
        #  * w przypadku kilku akcji o takiej samej ocenie - wybieramy losowo
        #  * gdy stosujemy technikę UCB pierwszeństwo mają akcje, które nie były jeszcze nigdy testowane
        # jak final będzie true to oznacza, że mamy ruch prrzeciwnika i mamy wybrać najlepszą akcję -
        # niby najczęściej odwiedzaną, ale nie przypadkiem z najlepszym wartościowaniem?

        if final: # choose action with most visits if planning phase finished
            max_visits = max(child.times_chosen for action, child in self.children.items())
            best_actions = [action for action, child in self.children.items() if child.times_chosen == max_visits]
            return random.choice(best_actions)

        ucb_estimates = dict()
        # calculate UCB estimates for each arm
        for action, child in self.children.items():
            # if child never visited choose it -> prevent division by 0
            if child.times_chosen == 0:
                return action
            else:
                exploit =  child.value # this is total_rewards / times_chosen - an estimate of the child’s true win-probability, based on all the rollouts done so far
                explore = self.c_coefficient * math.sqrt(math.log(self.times_chosen)/ child.times_chosen)
                ucb_estimates[action] = exploit + explore

        max_ucb = max(ucb_estimates.values())
        best_actions = [action for action, ucb_estimate in ucb_estimates.items() if ucb_estimate == max_ucb]

        return random.choice(best_actions)

    def expand(self) -> None:
        # after calling node.expand(), node.children becomes non‐empty (unless it was terminal)
        if not self.terminal and self.leaf:
            legal_moves = self.board.moves_for(self.current_player)
            if len(legal_moves) > 0:
                self.leaf = False
                oponent = self.current_player.flip()
                for move in legal_moves:
                    child_board = self.board.duplicate()
                    child_board.apply_move(self.current_player, move)
                    child = MCTSNode(child_board, oponent, self.c_coefficient)
                    child.parent = self
                    self.children[move] = child
            else:
                self.terminal = True

    def simulate(self) -> Colour: # this where "rollout" happens
        #  run a “rollout” or “playout” from the current board state down to a game‐over state,
        #  using purely random moves
        if not self.terminal:
            # w tym węźle rozgrywka nie zakończyła się, więc do ustalenia zwycięzcy potrzebna jest symulacja
            # podpowiedzi:
            #  * w tym celu najłatwiej uruchomić osobną, niezależną grę startującą z danego stanu planszy
            #  * by sumulacja przebiegała możliwe szybko wykonujemy ją z użyciem losowych agentów
            #  * po jej zakończeniu poznajemy i zwracamy zwycięzcę
            simulation_board = self.board.duplicate()
            current_player = self.current_player
            red_player = RandomPlayer()
            blue_player = RandomPlayer()
            game = Game(red_player, blue_player, simulation_board, current_player)
            game.run(verbose=False)
            return game.winner

        else:
            return self.current_player.flip() #  if it’s current_player’s turn and they have no moves, they lose and the opponent wins immediately

    def backpropagate(self, winner: Colour) -> None: # update rewards
        # należy zaktualizować drzewo - wiedząc, że przejście przez ten węzeł skończyło się wygraną danego gracza
        # podpowiedzi:
        #  * przede wszystkim należy zaktualizować licznik odwiedzeń (`self.times_chosen`)
        #  * poza tym, konieczna jest też korekta wartościowania (`self.value`)
        #     * siła korekty powinna zależeć od tego, które to z kolei odwiedziny danego węzła
        #     * uwaga - fakt, iż np. gracz czerwony wygrał partię ma inny wpływ na wartościowanie jego węzłów...
        #     * ...a inny na wartościowanie węzłów, w których ruch musiał wykonać jego przeciwnik
        #  * warto pamiętać, by po aktualizacji danych węzeł powiadomił o takiej konieczności również swojego rodzica

        # from winner node update upward - so update all parents until we reach parent = None
        current_node = self # current_node is "a pointer"

        while current_node.parent is not None:
            current_node.times_chosen += 1

            # IMPORTANT: dlatego nagroda w nodzie, w którym aktualny gracz nie jest wygranym, bo to jest węzeł,
            #  do którego wygrany poszedł - a więc on był dobrym wyborem i należy dać nagrodę
            if current_node.current_player != winner:
                reward = 1.0
            else:
                reward = 0.0

            # update value estimates by averaging accumulated awards over times visited
            current_node.value += (reward - current_node.value) / current_node.times_chosen

            # advance to the current node's parent
            current_node = current_node.parent

        # update how many times the root was visited
        current_node.times_chosen += 1

class MCTSPlayer(Player):
    def __init__(self, time_limit: float, c_coefficient: float):
        """
        An agent that uses Monte Carlo Tree Search to pick moves.
        When it’s asked to choose an action, it will run as many MCTS iterations as it can fit in self.time_limit,
        then pick the move that had the highest visited count.
        """
        self.time_limit: float = time_limit
        self.root_node: Optional[MCTSNode] = None
        self.c_coefficient: float = c_coefficient

    def choose_action(self, board: Board, current_player: Colour) -> tuple[int, int]:
        if self.root_node is None:
            self.root_node = MCTSNode(board.duplicate(), current_player, self.c_coefficient)

        # tha planning stage loop - the big "repeat while time remains"
        start_time = time.time()
        while True:
            self._mcts_iteration()  # does selection, expansion, simulation and backup

            current_time = time.time()
            elapsed_time = current_time - start_time
            if elapsed_time >= self.time_limit:
                break

        action = self.root_node.select(final=True)  # należy zmienić selekcje tak, by wybrała najlepszą akcję
        # podpowiedź: zamiast UCB wystarczy zwrócić akcję najczęściej odwiedzaną

        self._step_down(action)
        return action

    def register_opponent_action(self, action: tuple[int, int]) -> None:
        if self.root_node is not None:
            self.root_node.expand() # ensure that all children of the root node have been generated, so that you can step down into the correct one
            self._step_down(action) # make the child corresponding to the opponent’s move the new root

    def _mcts_iteration(self):
        node = self.root_node
        while not node.leaf:
            action = node.select()  # tree policy - UCB
            node = node.children[action]
        node.expand()   # got to the leaf - do expansion using tree policy
        winner = node.simulate()    # use rollout policy (fast - here: random) to do a MC rollout to check who wins
        node.backpropagate(winner) # update action values

    def _step_down(self, action: tuple[int, int]) -> None:
        """
        After a move is chosen, trim the tree so that the node corresponding to that move becomes the new root.
        """
        new_root = self.root_node.children[action]
        new_root.parent = None
        self.root_node = new_root


def main() -> None:
    red_wins = 0
    blue_wins = 0

    for _ in range(100):
        board = Board(7, 5)
        # red_player = RandomPlayer() # zastąp jednego z agentów wariantem MCTS
        red_player = MCTSPlayer(0.2, 0.5)
        blue_player = RandomPlayer()
        # game = Game(red_player, blue_player, board) # do zakomentowania, zalezy, kto ma zaczynać
        game = Game(blue_player, red_player, board)
        game.run(verbose=False)  # jeżeli nie chcesz czytać na konsoli zapisu partii, skorzystaj z `verbose=False`

        if game.winner == Colour.RED:
            red_wins += 1
        else:
            blue_wins += 1

    print(red_wins, blue_wins)  # jeżeli wszystko poszło dobrze, to agent MCTS powtarzalnie wygrywa z losowym -> tak


if __name__ == '__main__':
    main()  # jeżeli podstawowy eksperyment zakończył się sukcesem to sprawdź inne jego warianty
    # podpowiedź:
    #  * możesz zorganizować pojedynek agentów MCTS o różnych parametrach (np. czasie na wybór akcji)
    #  * możesz też zmienić rozmiar planszy lub skłonność do eksplorowania (`self.c_coefficient`)
    # zrobione w pliku experiments.py
