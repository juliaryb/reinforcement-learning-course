from typing import Optional

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import tensorflow_probability as tfp
from tqdm import tqdm


class ActorCriticController:
    def __init__(self, environment, learning_rate: float, discount_factor: float) -> None:
        self.environment = environment
        self.discount_factor: float = discount_factor
        self.state_dims: int = environment.observation_space.shape[0]
        self.action_dims: int = environment.action_space.n
        self.model: tf.keras.Model = self.create_actor_critic_model()
        self.optimizer: tf.keras.optimizers.Optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)  # przygotuj odpowiedni optymizator, pamiętaj o learning rate!
        self.log_action_probability: Optional[tf.Tensor] = None  # zmienna pomocnicza, przyda się do obliczania docelowej straty
        self.tape: Optional[tf.GradientTape] = None  # zmienna pomocnicza, związana z działaniem frameworku
        self.last_error_squared: float = 0.0  # zmienna używana do wizualizacji wyników
        self.critic_value: float = 0.0  # for keeping the tape recorded critic value

    # @staticmethod
    def create_actor_critic_model(self) -> tf.keras.Model:
        # przygotuj potrzebne warstwy sieci neuronowej o odpowiednich aktywacjach i rozmiarach
        # input layer - dims of state

        inputs = layers.Input(shape=(self.state_dims,))

        # first hidden layer
        x = layers.Dense(1024, activation="relu")(inputs)
        x = layers.LayerNormalization()(x)

        # second hidden
        x = layers.Dense(256, activation="relu")(x)
        x = layers.LayerNormalization()(x)

        # actor output layer
        actor_outputs = layers.Dense(self.action_dims, activation="softmax", name="policy_probs")(x)

        # critic output layer
        critic_outputs = layers.Dense(1, name="value")(x)

        return tf.keras.Model(inputs=inputs, outputs=[actor_outputs, critic_outputs])

    def choose_action(self, state: np.ndarray) -> int:
        state = self.format_state(state)  # przygotowanie stanu do formatu akceptowanego przez framework

        self.tape = tf.GradientTape(persistent=True)    # set to true so that it persists through the next with tape
        with self.tape:
            # wszystko co dzieje się w kontekście danej taśmy jest zapisywane i może posłużyć do późniejszego wyliczania pożądanych gradientów
            actor_probs, self.critic_value = self.model(state) # forward pass to get softmax activations (which represent the probability distributions for actions)
            dist = tfp.distributions.Categorical(probs=actor_probs[0])  # sample activations
            action = dist.sample()  # tu trzeba wybrać odpowiednią akcję korzystając z aktora - draws a discrete index i with Pr(i)=p_i
            self.log_action_probability = dist.log_prob(action)  # tu trzeba zapisać do późniejszego wykorzystania logarytm prawdopodobieństwa wybrania uprzednio wybranej akcji (będzie nam potrzebny by policzyć stratę aktora)
        return int(action)

    # noinspection PyTypeChecker
    def learn(self, state: np.ndarray, reward: float, new_state: np.ndarray, terminal: bool) -> None:
        state = self.format_state(state)
        new_state = self.format_state(new_state)

        with self.tape:  # to ta sama taśma, które użyliśmy już w fazie wybierania akcji
            # wszystko co dzieje się w kontekście danej taśmy jest zapisywane i może posłużyć do późniejszego wyliczania pożądanych gradientów
            _, new_critic_value = self.model(new_state)
            if not terminal:
                td_return = reward + self.discount_factor * tf.stop_gradient(new_critic_value) # stop_gradient treats the argument as fixed (doesn't take it into account when calculating gradients)
            else:
                td_return = reward

            delta = td_return - self.critic_value  # tu trzeba obliczyć błąd wartościowania aktualnego krytyka (delta ^ 2)

            loss_critic = tf.square(delta)
            loss_actor = -1 * tf.stop_gradient(delta) * self.log_action_probability

            loss = loss_critic + loss_actor  # tu trzeba obliczyć sumę strat dla aktora i krytyka

        self.last_error_squared = loss_critic.numpy().item()
        gradients = self.tape.gradient(loss, self.model.trainable_weights)  # tu obliczamy gradienty po wagach z naszej straty, pomagają w tym informacje zapisane na taśmie
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_weights))  # tutaj zmieniamy wagi modelu wykonując krok po gradiencie w kierunku minimalizacji straty

        del self.tape

    @staticmethod
    def format_state(state: np.ndarray) -> np.ndarray:
        return np.reshape(state, (1, state.size))


def main() -> None:
    environment = gym.make('CartPole-v1', render_mode='human')  # zamień na gym.make('LunarLander-v2', render_mode='human') by zająć się lądownikiem
    # zmień lub usuń render_mode, by przeprowadzić trening bez wizualizacji środowiska
    controller = ActorCriticController(environment, 0.00001, 0.99)

    past_rewards = []
    past_errors = []
    for i_episode in tqdm(range(2000)):  # tu decydujemy o liczbie epizodów
        done = False
        truncated = False
        state, info = environment.reset()
        reward_sum = 0.0
        errors_history = []

        while not done and not truncated:
            environment.render()  # tą linijkę możemy wykomentować, jeżeli nie chcemy mieć wizualizacji na żywo

            action = controller.choose_action(state)
            new_state, reward, done, trunctated, info = environment.step(action)
            controller.learn(state, reward, new_state, done)
            state = new_state
            reward_sum += reward
            errors_history.append(controller.last_error_squared)

        past_rewards.append(reward_sum)
        past_errors.append(np.mean(errors_history))

        window_size = 50  # tutaj o rozmiarze okienka od średniej kroczącej
        if i_episode % 25 == 0:  # tutaj o częstotliwości zrzucania wykresów
            if len(past_rewards) >= window_size:
                fig, axs = plt.subplots(2)
                axs[0].plot(
                    [np.mean(past_errors[i:i + window_size]) for i in range(len(past_errors) - window_size)],
                    'tab:red',
                )
                axs[0].set_title('mean squared error')
                axs[1].plot(
                    [np.mean(past_rewards[i:i+window_size]) for i in range(len(past_rewards) - window_size)],
                    'tab:green',
                )
                axs[1].set_title('sum of rewards')
            plt.savefig(f'plots/learning_{i_episode}.png')
            plt.clf()

    environment.close()
    controller.model.save("final.model")  # tu zapisujemy model


if __name__ == '__main__':
    main()
