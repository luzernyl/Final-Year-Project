# Comparative Study of Solvers in Generative Diffusion Models

Code for final year paper (project)

## Abstract 
The study of diffusion models has gained popularity in the field of generative models, with models showing promising results in generating high-quality images. In this thesis, we first provide an overview of the development of diffusion models its applications in generative models. We then discuss the importance of Stochastic Differential Equation (SDE) solvers in simulating the dynamics of diffusion models, as well as the mathematical model behind SDE-based diffusion models. 

The thesis provides a comprehensive overview of the mathematical principles behind SDE solvers, including the concept of weak and strong convergence, and how these principles are applied in different SDE solvers. The thesis then proceeds to evaluate the performance of each solver by comparing the sample quality produced by each method in three different evaluation metrics: Inception Score (IS), Fréchet Inception Distance (FID), and Kernel Inception Distance (KID). The results demonstrate that the SRK2 solver consistently outperforms the other solvers in terms of sample quality across all evaluation metrics, while also maintaining reasonable sampling times. However, the Euler-Maruyama solver, which has the shortest sampling time, is not far behind and is a good option when time is a constraint. We also found that longer sampling time does not necessarily mean better sample quality. Overall, this thesis provides insights into the importance of SDE solvers in generating high-quality samples in diffusion models and can help researchers choose the most suitable solver for their specific application.

Links for data and evaluation results :
Dataset : https://drive.google.com/file/d/12PXcfqmdEQ0CKTr6jFVZAr2JoTeWKQl8/view?usp=drive_link
Checkpoint : https://drive.google.com/file/d/1eIv7z4nLtMW01YE1KDjtbS297vEJt9p7/view?usp=drive_link'
CIFAR-10 statistics : https://drive.google.com/file/d/1CfFMlySNmbpfW5P1rzz9NR63ALgjrtSR/view?usp=drive_link
Evaluation Results : https://drive.google.com/drive/folders/1JPNfO5izL0nRBBhHnjzaTUxOHLqPbPY5?usp=sharing
