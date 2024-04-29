Rapport projet Pacman renforcement 

 

Exercice 1 – Tabular Q-Leraning : 

Dans des petits niveaux tel que celui du fichier level0.lay, on remarque que la stratégie TabularQlearning fonctionne très bien. Le pacman apprend très rapidement comment gagner la partie.  

 ![resultat test generation 1](file:///home/etud/apprentissage/projetPacman/PacmanApprentissage/images/gene1_exo1.png)

 ![resultat test generation 40](file:///home/etud/apprentissage/projetPacman/PacmanApprentissage/images/gene40_exo1.png)

 Lorsqu’on augment de niveaux avec le level1.lay, le pacman commence à avoir plus de difficulté. Cela se remarque d’autant plus dans le level2.lay car il y a plus d’état à visiter. Le pacman s’améliore mais cela prend énormément de temps. Ce n’est qu’au bout d’une vingtaines de générations qu’on observe la moyenne globale des récompenses passer au positive. On constate également, que la moyenne globale des récompenses avec le mode test sont mieux que celui de train. Au bout de plus de 100 générations les résultats s’améliorent largement, mais celles-ci restent relativement faibles.  

Dans l’image ci-dessous on peut constater les résultats au bout de 149 génération. 

  ![resultat test](file:///home/etud/apprentissage/projetPacman/PacmanApprentissage/images/qlearning.png)



Exercice 2 – Approximate Q-Learning : 

feactures proposés : 

    Le nombre de coup à jouer pour atteindre la pacgomme la plus proche en jouant l’action a à l’état t 

    Booléen si le pacman peut manger une pacgomme en effectuant l’action a à l’état t 

    Présence ou non d’un fantôme aux emplacements adjacentes (nord, sud, est, ouest, nord-est, nord-ouest, sud-est, sud-ouest) en jouant l’action a à l’état t 

    Présence ou non d’un mur lorsqu’on joue l’action a à l’état t 
