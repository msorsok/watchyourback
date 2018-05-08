import numpy as np
import referee_trainer
def main():
    p1 = 0
    p2 = 0
    draws = 0
    player1 = "player_1layer"
    player1 = "playerRandom"
    #player1= "playerLearner"
    player2 = "player"
    for x in range(1, 21):
        if np.random.randint(0,2) == 1:
            winner = referee_trainer.main(player1, player2)
            if (winner == "B"):
                print(x, player2)
                p2+=1
            elif (winner == "W"):
                print(x, player1)
                p1+=1
            else:
                print(x, "Draw")
                draws+=1
        else:
            winner = referee_trainer.main(player2, player1)
            if (winner == "W"):
                print(x, player2)
                p2+=1
            elif (winner == "B"):
                print(x, player1)
                p1+=1
            else:
                print(x, "Draw")
                draws+=1

    print(player1, ": ", p1)
    print(player2, ": ", p2)
    print("Draws: ", draws)

main()