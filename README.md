## U2I Matrix
* Raw weblog.

  ($u_1$, $i_1$), ($u_1$, $i_4$), ($u_1$, $i_4$), ($u_2$, $i_3$), ($u_2$, $i_5$), ($u_3$, $i_2$)

* User-item matrix **R**, which is derived from the raw log.

  ![image](https://github.com/user-attachments/assets/b3ead729-4285-4237-9be2-b9dd4ae9c321)

## I2I Matrix
* Item-item co-occurrence matrix is calculated from matrix transpose * itself.
* I2I matrix (i2i_rating) is also able to be computed by self-join SQL.

  ![image](https://github.com/user-attachments/assets/8fc52831-b263-4580-ba83-d8098a830cfe)

## Matrix Factorization
#### Concept
![image](https://github.com/user-attachments/assets/8b754cc6-6f2f-4d38-8081-d19a1ac2ce82)

#### Why we needs Matrix Factorization?
* Most of similarity based methods are filters, e.g. association rule, LLR, cosine similarity, and etc., which filter out unqualified items from the co-occurrence list.  
  e.g. $i_1$ => $[i_4]$ or none, $i_3$ => $[i_5]$, or none
* In matrix factorization, item-item similarity can be generated even they aren't co-occurrence in training set. i.e. infer other indicators from few observation.  
  e.g. $i_2$ => $[i_1, i_3, i_4, i_5]$

#### Input 
* Item-item co-occurrence matrix (i2i_rating) with either standard unit deviation or min-max normalization.
* ~3.3M item-item records with ~222K items (1 week data).
* In order to ensure data quality, the long and short transactions are removed, i.e. `< 3` or `> 50`
* training set = testing set, for validation conveniently.

#### [Algorithm - Spark ALS](https://spark.apache.org/docs/latest/mllib-collaborative-filtering.html)
For more detail, you are highly recommended to read the the article [Matrix Factorization Techniques for Recommender Systems (MFTRS)](https://datajobs.com/data-science-repo/Recommender-Systems-%5BNetflix%5D.pdf). And all equations that are described here also refers from MFTRS.

![image](https://github.com/user-attachments/assets/155ce7d5-ce9f-4853-95d9-c3a75df05b02)

where,

$K$ is the set of all training $(u,i)$ pairs.

$r_{ui}$ is the known rating from training data.

$\hat{r_{ui}} = p^T_uq_i$

$p_u$ is an user latent vector, in our case, we could thought as $p_i$.

$q_i$ is an item latent vector, in our case, we could thoght as $p^T_i$.

Considers the confidence level $c_{ui}$ of the implicit feedback and refines ![](http://latex.codecogs.com/png.latex?%5Chat%7Br_%7Bui%7D%7D) by adding biases ![](http://latex.codecogs.com/png.latex?%5Chat%7Br_%7Bui%7D%7D%3D%5Cmu%20&plus;%20b_i%20&plus;%20b_u%20&plus;%20p_u%5ET%20q_i), we could get the final equation as below.

![](https://latex.codecogs.com/png.latex?%5Cmin_%7Bp%5E*%2Cq%5E*%2Cb%5E*%7D%5Csum_%7B%28u%2Ci%29%5Cin%20k%7Dc_%7Bui%7D%28r_%7Bui%7D-u-b_u-b_i-p_u%5ETq_i%29%5E2&plus;%5Clambda%20%28%5Cleft%5C%7C%20p_u%20%5Cright%5C%7C%5E2&plus;%5Cleft%5C%7C%20q_i%20%5Cright%5C%7C%5E2%20&plus;%20b_u%5E2%20&plus;%20b_i%5E2%29)

```
val model = new ALS()
    .setImplicitPrefs(implicitPrefs)
    .setRank(rank)
    .setAlpha(alpha)
    .setIterations(iterations)
    .setLambda(lambda)
    .run(i2i_rating) 
```
* [Parameters](https://spark.apache.org/docs/latest/api/scala/index.html#org.apache.spark.mllib.recommendation.ALS)
  * **implicitPrefs** -> `true` to use implicit preference
  * **rank**  - `> 100` with the better result 
  * **alpha** - `> 1, [1, 1e1, 1e2, 1e3, 1e4]`
  * lambda - `[1e-3, 1e-2, 1e-1, 1, 2]`, the regularization parameter
    * the higher the lambda, the lower the over-fitting but the lose accuracy (greater the bias).
    * small RMSE is always made up of the higher rank and lower lambda.
  * iterations - `[10, 20]`, higher is better and greater than 20 always introduces `java.lang.StackOverflowError`

* Feature Normalization
  * [StandardScaler](https://spark.apache.org/docs/latest/ml-features.html#standardscaler)
  * [MinMaxScaler](https://spark.apache.org/docs/latest/ml-features.html#minmaxscaler), [0, 1]

## [Experimental Result](https://docs.google.com/spreadsheets/d/13mQ4leXf2mS52EcTPrnMedPLN6KlBSYgmkq5-etm064/edit#gid=525028610)
![](https://drive.google.com/uc?id=0B78KhWqVkVmtcWlBaVI3Zk5CdFU)
According experiments, implicit preference with standard unit deviation normalization training data under higher rank, higher alpha and low lambda always get the better result.

There is an interesting observation from our experiments. 
The lowest RMSE doesn't always stand for best recommendation list.
So far, the parameter set in terms of the most meaningful recommendation list should be verified manually.

## Reference
* [MATRIX FACTORIZATION TECHNIQUES FOR RECOMMENDER SYSTEMS](https://datajobs.com/data-science-repo/Recommender-Systems-%5BNetflix%5D.pdf)
* [Using Machine Learning on Compute Engine to Make Product Recommendations](https://cloud.google.com/solutions/recommendations-using-machine-learning-on-compute-engine)
* [Movie-recommendation, AMP Camp](http://ampcamp.berkeley.edu/big-data-mini-course/movie-recommendation-with-mllib.html)
* [Matrix Factorization with Tensorflow](http://katbailey.github.io/post/matrix-factorization-with-tensorflow/)
* [Spotify - Music Recommendations at Scale with Spark](https://www.slideshare.net/MrChrisJohnson/music-recommendations-at-scale-with-spark)
