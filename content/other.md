<link rel="stylesheet" 
type="text/css" 
href="/theme/css/md-style.css">

<p class = "body">
If you're a working data scientist then you have likely heard of Variational Autoencoders and their interesting applications in data generation. After assiduous hours of studying this model, I remember its end-to-end derivation leaving me in a state of theoretical bliss. Reason being, it brings hierarchical graphical models, variational inference and deep learning together in this elegant way that can be traced back to their respective roots.
</p>

<h1 class="head">1. Introduction </h1>
<p class="body">
Like all 
<a href="https://openai.com/blog/generative-models/ "style="color:#FF7D33">
generative models </a>, VAEs are concerned with learning a valid probability distribution from which our dataset was <i>generated</i> from. They do this by learning 
a set of latent variables, 
<span class = "math">\( \boldsymbol{z} \) </span>,
$\boldsymbol{z}$
which are essentially the underlining features that best explain our observed data
<span class = "math">\( \boldsymbol{x} \) </span>.
We can then condition on these variables and sample from the distribution, 
<span class = "math">
\( 
\boldsymbol{x}_{new} \sim 
p(\boldsymbol{x}_{observed}|\boldsymbol{z}) \)
</span>, 
to generate a new data point. This distribution happens to be tractable 
but using the posterior,
<span class = "math">
\( 
\boldsymbol{z} \sim 
p(\boldsymbol{z}|\boldsymbol{x}_{observed}) \)
</span>,
is the distribution that has plagued machine learning since the beginning of time.
Furthermore, it is challenging to learn the parameters of both distributions in an end-to-end fashion. developed the Wake-sleep algorithm which was able to learn all parameters to this exact same problem via two objective functions and back-propagation. While this algorithm <i> masked </i> the problem at hand, even Geoffrey Hinton 
<a href="https://www.youtube.com/watch?v=VKpc_z7b9I0" style="color:#FF7D33"> 
acknowledged </a>, and I quote, <i> "the basic idea behind these variational methods sounds crazy." </i> Luckily and deep generative models have really been in resurgence since! 



<h class = "head"> 2. Problem Scenario </h>
<p class="body">
Assume our 
</p>

<figure>
  <img class="fig" src="/images/vae/pgm.png" width="50%">
  <figcaption class="figcaption"> 
  Fig 1. Graphical model in concern 
  </figcaption>
</figure>

<p class = "body">
Our goal is to use maximum likelihood estimation to learn the parameters, <span class="math"> \(\boldsymbol{\theta}\)
</span>, of the following distribution:
</p>

<p class = "formula">
$$
p(\boldsymbol{x}) =  \int p_{\boldsymbol{\theta}}(\boldsymbol{x|z})
p_{\boldsymbol{\theta}}(\boldsymbol{z}) \ d\boldsymbol{z}
$$
</p>

<p class = "body">
There are some underlining takeaways from the above distribution:
</p>


<ul class="body">
<li> The probability mass or density functions on the right can be the output of a neural network (assuming it returns a valid probability distribution) or we can use a neural network to generate the parameters of a more traditional distribution, say a Gaussian (which is a frequently used approach). </li>

<li> Because we have no data on the latent variables, they need to be integrated out during MLE. Unfortunately, this tends to be intractable because our model assumes <span class="math"> \(z_i \not\!\perp\!\!\!\perp z_j | \boldsymbol{x} \ \forall \ i \neq j \) </span> which requires us to sum out all combinations our latent variables can take on.
</li>
</ul>
<p class="body">
Since our current distribution is intractable, we must resort to variational inference to learn 
<span class = "math">\( \boldsymbol{\theta} \). 
</span>
</p>
<h1 class = "head"> 3. Variational Inference </h1>
<p class="body">
If we recall, MLE is a simple optimization problem which returns a set of parameters that maximizes the likelihood that our data came from some distribution of our choice. Our objective function is the log likelihood and variational inference simply replaces this function with a lower bound function (ELBO or variational lower bound) that is  <i>tractable</i>. So, as we learn a set of parameters that maximizes the likelihood of our lower bound we are also increasing the likelihood of our original distribution (this is the definition of a lower bound).
</p>
<h2 class="subhead"> 3.1 Deriving the ELBO </h2>
<p class="body">
The derivation of the lower bound is a mere byproduct of  
<a url = "https://www.probabilitycourse.com/chapter6/6_2_5_jensen's_inequality.php" 
style="color:#FF7D33">
Jensen's Inequality</a>
which simply states: for any concave function, <span class = "math">\(g,\) </span>
we have <span class="math"> \( g(\mathbb{E}[u]) \geq \mathbb{E}[g(u)]\). 
</span> So, we layout the following:
</p>
<p class="formula">
$$
\begin{aligned}
    \log p(\boldsymbol{x}) &= \log \int p_{\boldsymbol{\theta}}(\boldsymbol{x|z})
    p_{\boldsymbol{\theta}}(\boldsymbol{z}) \ d\boldsymbol{z} \\
    &= \log \int p_{\boldsymbol{\theta}}(\boldsymbol{x},\boldsymbol{z}) \ d\boldsymbol{z} \\
    &= \log \int q_{\boldsymbol{\phi}}(\boldsymbol{z}|\boldsymbol{x})
    \frac{p_{\boldsymbol{\theta}}(\boldsymbol{x},\boldsymbol{z})}
    {q_{\boldsymbol{\phi}}(\boldsymbol{z}|\boldsymbol{x})} \ d\boldsymbol{z}
\end{aligned}
$$
</p>
<p class="body">
Ignoring <span class="math">\(q_{\boldsymbol{\phi}} \)</span> for a second, all we have done thus far is rewritten our log likelihood function in a <i> clever </i> way. I emphasis clever because we can now invoke Jenson's Inequality where:
</p>
<p class="formula">
$$
\begin{aligned}
    u &= \frac{p_{\boldsymbol{\theta}}(\boldsymbol{x},\boldsymbol{z})}
    {q_{\boldsymbol{\phi}}(\boldsymbol{z}|\boldsymbol{x})}  \\
    \mathbb{E}_{\boldsymbol{z}\sim q_{\boldsymbol{\phi}}(\boldsymbol{z|x})} &=
    \int q_{\boldsymbol{\phi}}(\boldsymbol{z}|\boldsymbol{x}) \ d\boldsymbol{z} \\
    g(\cdot) &= \log(\cdot) \\
     \log p(\boldsymbol{x}) &= \log 
     \mathbb{E}_{\boldsymbol{z}\sim q_{\boldsymbol{\phi}}(\boldsymbol{z|x})}
     \bigg[ \frac{p_{\boldsymbol{\theta}}(\boldsymbol{x},\boldsymbol{z})}
     {q_{\boldsymbol{\phi}}(\boldsymbol{z}|\boldsymbol{x})}\bigg] \\
     &\geq  \mathbb{E}_{\boldsymbol{z}\sim q_{\boldsymbol{\phi}}(\boldsymbol{z|x})}
     \bigg[\log 
     \frac{p_{\boldsymbol{\theta}}(\boldsymbol{x},\boldsymbol{z})}
     {q_{\boldsymbol{\phi}}(\boldsymbol{z}|\boldsymbol{x})}
     \bigg] = \mathcal{L}(\boldsymbol{x}) && \textit{this is the ELBO} 
\end{aligned}
$$
</p>
<p class="body">
Going forward, we will use <span class = "math"> \(\mathcal{L}(\boldsymbol{x})\) </span> as our objective function of concern where <span class="math"> \(q_{\boldsymbol{\phi}}
(\boldsymbol{z}|\boldsymbol{x})\) </span> is essentially a surrogate distribution for <span class="math"> \(p_{\boldsymbol{\theta}}(\boldsymbol{z|x}) \)</span>. Now, <span class="math"> \(q_{\boldsymbol{\phi}}(\boldsymbol{z}|\boldsymbol{x})\) </span> will be some variation of a deep model, parameterized by <span class="math"> \(\boldsymbol{\phi}\) </span>,
where we now assume <span class="math"> \(z_i \perp z_j | \boldsymbol{x} \ \forall \ i \neq j \) </span>, thus, making it a much more tractable distribution.  
</p>

<h1 class="subhead"> 3.1.1 Making More Sense of the ELBO </h1>
<p class="body">
Understanding some variations of the ELBO really sheds light on the underlining intuitions of Variational Inference. For starters, we can rewrite the ELBO in the following way:
</p>
<p class="formula">
$$
\begin{aligned}
\mathcal{L}(\boldsymbol{x}) &=  
\mathbb{E}_{\boldsymbol{z}\sim q_{\boldsymbol{\phi}}(\boldsymbol{z|x})}
\big[\log p_{\boldsymbol{\theta}}(\boldsymbol{z|x})
 p_{\boldsymbol{\theta}}(\boldsymbol{x})\big]
- \mathbb{E}_{\boldsymbol{z}\sim q_{\boldsymbol{\phi}}(\boldsymbol{z|x})}
\big[\log q_{\boldsymbol{\phi}}(\boldsymbol{z|x})\big] \\
&=  
\mathbb{E}_{\boldsymbol{z}\sim q_{\boldsymbol{\phi}}(\boldsymbol{z|x})}
\big[\log p_{\boldsymbol{\theta}}(\boldsymbol{z|x})
+ \log p_{\boldsymbol{\theta}}(\boldsymbol{x})\big]
- \mathbb{E}_{\boldsymbol{z}\sim q_{\boldsymbol{\phi}}(\boldsymbol{z|x})}
\big[\log q_{\boldsymbol{\phi}}(\boldsymbol{z|x})\big] \\
&= 
\mathbb{E}_{\boldsymbol{z}\sim q_{\boldsymbol{\phi}}(\boldsymbol{z|x})}
\big[\log p_{\boldsymbol{\theta}}(\boldsymbol{z|x})\big]
+ \log p_{\boldsymbol{\theta}}(\boldsymbol{x})
\mathbb{E}_{\boldsymbol{z}\sim q_{\boldsymbol{\phi}}(\boldsymbol{z|x})}[] 
- \mathbb{E}_{\boldsymbol{z}\sim q_{\boldsymbol{\phi}}(\boldsymbol{z|x})}
\big[\log q_{\boldsymbol{\phi}}(\boldsymbol{z|x})\big] \\
&= 
 \log p_{\boldsymbol{\theta}}(\boldsymbol{x}) - 
 \text{D}_{\text{KL}}(q_{\boldsymbol{\phi}}(\boldsymbol{z|x}) || 
p_{\boldsymbol{\theta}}(\boldsymbol{z|x}) )
\end{aligned}
$$
</p>
<p class="body"> 
Note, the second expectation on the third line does not touch <span class="math">
\(p_{\boldsymbol{\theta}}(\boldsymbol{x})\) </span>, so that expectation evaluates to 1. Furthermore, we should notice in the last line <a href = "https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence" style="color:#FF7D33"> KL Divergence</a> has made an appearance-which is a non-negative number that quantifies the distance between two distributions; where 0 is returned if the two distributions under concern are identical.
</p>

<br>
<figure>
  <img class="fig" src="/images/vae/kl-divergence.png" width="100%">
  <figcaption class="figcaption"> 
  Fig 2. Depiction of KL Divergence
  </figcaption>
</figure>
<br>

<p class = "body">
So, maximizing the lower bound is actually maximizing our original log probability and subtracting the distance between the intractable distribution and the surrogate distribution. Theoretically, we could set <span class = "math"> \(
q_{\boldsymbol{\phi}}(\boldsymbol{z|x})  = q_{\boldsymbol{\theta}}(\boldsymbol{z|x})\)
</span> and our lower bound would be tight!
</p>

<h1 class="head"> 4. Relationship to Autoencoders </h1>

<p class = "body">
Thus far, all we have done is explore the variational component of VAEs by understanding 
the theoretical properties of the lower bound. Now, we will begin to understand the 
learning process of VAEs and their relationship to autoencoders. Let's use derive one more flavor of the lower bound which is amendable for learning: 
</p> 

<p class="formula">
$$
\begin{aligned}
\mathcal{L}(\boldsymbol{x}) &= 
 \log p_{\boldsymbol{\theta}}(\boldsymbol{x}) - 
 \text{D}_{\text{KL}}(q_{\boldsymbol{\phi}}(\boldsymbol{z|x}) || 
p_{\boldsymbol{\theta}}(\boldsymbol{z|x})) \\
&=
\log p_{\boldsymbol{\theta}}(\boldsymbol{x}) -
\mathbb{E}_{\boldsymbol{z}\sim q_{\boldsymbol{\phi}}(\boldsymbol{z|x})}
\bigg[\log\frac{q_{\boldsymbol{\theta}}(\boldsymbol{z}|\boldsymbol{x})}
{p_{\boldsymbol{\theta}}(\boldsymbol{z|x})}\bigg] \\
&=
\log p_{\boldsymbol{\theta}}(\boldsymbol{x}) - 
\mathbb{E}_{\boldsymbol{z}\sim q_{\boldsymbol{\phi}}(\boldsymbol{z|x})} \bigg[
\log\frac{q_{\boldsymbol{\theta}}(\boldsymbol{z}|\boldsymbol{x})} 
{\frac{p_{\boldsymbol{\theta}}(\boldsymbol{x|z})p_{\boldsymbol{\theta}}(\boldsymbol{z})}
{p_{\boldsymbol{\theta}}(\boldsymbol{x})}}
\bigg] \\
&= 
\log p_{\boldsymbol{\theta}}(\boldsymbol{x}) - 
\mathbb{E}_{\boldsymbol{z}\sim q_{\boldsymbol{\phi}}(\boldsymbol{z|x})} \bigg[
\log q_{\boldsymbol{\phi}}(\boldsymbol{z|x}) + 
\log p_{\boldsymbol{\theta}}(\boldsymbol{x}) - 
\log p_{\boldsymbol{\theta}}(\boldsymbol{x|z}) - 
\log p_{\boldsymbol{\theta}}(\boldsymbol{z})
\bigg] \\
&= 
\log p_{\boldsymbol{\theta}}(\boldsymbol{x}) - 
\log p_{\boldsymbol{\theta}}(\boldsymbol{x})
\mathbb{E}_{\boldsymbol{z}\sim q_{\boldsymbol{\phi}}(\boldsymbol{z|x})}[] -
\mathbb{E}_{\boldsymbol{z}\sim q_{\boldsymbol{\phi}}(\boldsymbol{z|x})}\bigg[
\log q_{\boldsymbol{\phi}}(\boldsymbol{z|x}) -
\log p_{\boldsymbol{\theta}}(\boldsymbol{x|z}) - 
\log p_{\boldsymbol{\theta}}(\boldsymbol{z})
\bigg] \\
&=
-\mathbb{E}_{\boldsymbol{z}\sim q_{\boldsymbol{\phi}}(\boldsymbol{z|x})}
[\log q_{\boldsymbol{\phi}}(\boldsymbol{z|x}) -
\log p_{\boldsymbol{\theta}}(\boldsymbol{z})] +
\mathbb{E}_{\boldsymbol{z}\sim q_{\boldsymbol{\phi}}(\boldsymbol{z|x})}
[\log p_{\boldsymbol{\theta}}(\boldsymbol{x|z})] \\
&=
\underbrace{
\mathbb{E}_{\boldsymbol{z}\sim q_{\boldsymbol{\phi}}(\boldsymbol{z|x})}
[\log p_{\boldsymbol{\theta}}(\boldsymbol{x|z})]}_
{\mathcal{L}_{\text{recon}}(\boldsymbol{x})} -
\underbrace{
\text{D}_{\text{KL}}(q_{\boldsymbol{\phi}}(\boldsymbol{z|x})
||p_{\boldsymbol{\theta}}(\boldsymbol{z}))}_
{\mathcal{L}_{\text{reg}}(\boldsymbol{x})}
\end{aligned}
$$
</p>