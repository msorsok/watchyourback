import numpy as np
import referee
def main():
    p1 = 0
    p2 = 0
    draws = 0
    player1 = "playerRandom"
    player1= "playerLearner"
    player2 = "player"
    for x in range(20):
        if np.random.randint(0,2) == 1:
            winner = referee.main(player1, player2)
            if (winner == "B"):
                print("Player2")
                p2+=1
            elif (winner == "W"):
                print("Player1")
                p1+=1
            else:
                print("Draw")
                draws+=1
        else:
            winner = referee.main(player2, player1)
            if (winner == "W"):
                print("Player2")
                p2+=1
            elif (winner == "B"):
                print("Player1")
                p1+=1
            else:
                print("Draw")
                draws+=1

    print("Player1: ", p1)
    print("Player2: ", p2)
    print("Draws: ", draws)

main()