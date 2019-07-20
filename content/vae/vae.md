---
Title: Variational Autoencoders
Date: 07/20/2019
Category: Probabilistic Graphical Models
Slug: Variational Autoencoders
Summary: TEXT TEXT
---
<link rel="stylesheet" 
type="text/css" 
href="/theme/css/md-style.css">

<h1 class="head">1. Introduction </h1>
<p class="body">
Like all 
<a href="https://openai.com/blog/generative-models/" style="color:#FF7D33">
generative models</a>, VAEs are concerned with learning a valid probability distribution from which our dataset was <i>generated</i> from. They do this by learning a set of latent variables, 

<span class="math">\(\boldsymbol{z}\)</span>,


which are essentially the underlining features that best explain our observed data,

<span class = "math">\(\boldsymbol{x}\)</span>.

We can then condition on these variables and sample from the distribution, 
<span class = "math">
\( 
\boldsymbol{x}_{new} \sim 
p(\boldsymbol{x}_{observed}|\boldsymbol{z}) \)</span>, 

to generate a new data point. This distribution happens to be tractable 
but using the posterior,

<span class = "math">
\( 
\boldsymbol{z} \sim 
p(\boldsymbol{z}|\boldsymbol{x}_{observed}) \)</span>,

is the distribution that has plagued machine learning since the beginning of time.
Furthermore, it is challenging to learn the parameters of both distributions in an end-to-end fashion. [@@wakeSleep] developed the Wake-sleep algorithm which was able to learn all parameters to this exact problem via two objective functions and back-propagation. While this algorithm <i> masked </i> the problem at hand, even Geoffrey Hinton 
<a href="https://www.youtube.com/watch?v=VKpc_z7b9I0" style="color:#FF7D33"> 
acknowledged</a>, and I quote, <b><i> "the basic idea behind these variational methods sounds crazy." </i></b> Luckily, [@@vae] offered a better solution using the <i>
reparameterization trick </i> and deep generative models have really been in resurgence since! 
</p>

<pre>


</pre>

<h1 class="head"> 2. Problem Scenario </h1>
<p class="body">

Let us begin by assuming two layers of latent variables where
<span class="math">
\(\boldsymbol{z}=\{\boldsymbol{z}^1,\boldsymbol{z}^2\}\)</span>.
As a result, we have the following graphical model:
<br>
</p>

<figure>
  <img class="fig" src="/images/vae/pgm.png" width="50%">
  <figcaption class="figcaption"> 
  Fig 1. Graphical model in concern  
  </figcaption>
</figure>
</p>

<p class="body">
<br>
It should be noted [@@iwae] proposed this architecture which is a mere generalization of the original architecture used in [@@vae]. The goal is to use maximum likelihood estimation to learn the parameters, 
<span class="math">\(\boldsymbol{\theta}\)</span>, of the following distribution:
</p>

<div class="math">
<br>
<p class="formula">
$$
p(\boldsymbol{x}) =  \int p_{\boldsymbol{\theta}}(\boldsymbol{x|z})
p_{\boldsymbol{\theta}}(\boldsymbol{z}) \ d\boldsymbol{z}
$$
</p>
</div>

<p class = "body">
<br>
There are some underlining takeaways from the above distribution:
</p>

<ul class="body">
<br>
<li> It tends to be intractable because our model assumes 

<span class="math"> \(z_i^l \not\!\perp\!\!\!\perp z_j^l | \ \boldsymbol{x} \ \cup \
\boldsymbol{z^{l-1}} \ 
\forall \ i \neq j,l \) </span> 

<a href="https://sailinglab.github.io/pgm-spring-2019/notes/lecture-02/#active_trail" style="color:#FF7D33">
(explaining away)</a>.
This requires integration over <b>all</b> combinations our latent variables can take on.
</li>

<li> The probability density functions on the right tends to a traditional distribution, say a Gaussian, but is parameterized by some deep network. </li>
<br>
</ul>

<p class="body">
Since our current distribution is intractable, we must resort to variational inference to learn 

<span class = "math">\( \boldsymbol{\theta} \).</span>

</p>

<pre>


</pre>
<h1 class = "head"> 3. Variational Inference </h1>
<p class="body">

If we recall, MLE is a simple optimization problem which returns a set of parameters that maximizes the likelihood that our data came from some distribution of our choice. Our objective function is the log likelihood and variational inference simply replaces this function with a lower bound function (evidence lower bound or ELBO) that is  <b>tractable</b>. So, as we learn a set of parameters that maximizes the likelihood of our lower bound we are also increasing the likelihood of our original distribution (this is the definition of a lower bound).
</p>

<h1 class="subhead"> 3.1 Deriving the ELBO </h1>
<p class="body">

The derivation of the lower bound is a mere byproduct of  
<a href = "https://www.probabilitycourse.com/chapter6/6_2_5_jensen's_inequality.php" style="color:#FF7D33">Jensen's Inequality</a>

which simply states: for any concave function, 

<span class = "math">\(f,\) </span>
we have <span class="math"> \( f(\mathbb{E}[u]) \geq \mathbb{E}[f(u)]\). </span> 

So, we layout the following:
</p>

<div class="math">
<p class="formula">
<br>
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
<br>
</div>

<p class="body">

Ignoring 

<span class="math">\(q_{\boldsymbol{\phi}} \)</span> 

for a second, all we have done thus far is rewritten our log likelihood function in a <i>clever</i> way. I emphasis clever because we can now invoke Jensen's Inequality where:

</p>

<div class="math">
<p class="formula">
<br>
$$
\begin{aligned}
    u &= \frac{p_{\boldsymbol{\theta}}(\boldsymbol{x},\boldsymbol{z})}
    {q_{\boldsymbol{\phi}}(\boldsymbol{z}|\boldsymbol{x})}  \\
    \mathbb{E}_{\boldsymbol{z}\sim q_{\boldsymbol{\phi}}(\boldsymbol{z|x})} &=
    \int q_{\boldsymbol{\phi}}(\boldsymbol{z}|\boldsymbol{x}) \ d\boldsymbol{z} \\
    f(\cdot) &= \log(\cdot) \\
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
<br>
</p>
</div>

<p class="body">

Going forward, we will use <span class = "math"> \(\mathcal{L}(\boldsymbol{x})\) </span> as our objective function of concern where 

<span class="math"> \(q_{\boldsymbol{\phi}}
(\boldsymbol{z}|\boldsymbol{x})\) </span> 

is essentially a surrogate distribution for 

<span class="math"> \(p_{\boldsymbol{\theta}}(\boldsymbol{z|x}) \)</span>. 

Now, we define the following for
<span class="math"> 
\(q_{\boldsymbol{\phi}}(\boldsymbol{z}|\boldsymbol{x})\)</span>: 

</p>


<div class="formula">
<br>
$$
\begin{aligned}
q_{\boldsymbol{\phi}}(\boldsymbol{z}|\boldsymbol{x}) &= 
q_{\boldsymbol{\phi}}(\boldsymbol{z}^2|
\boldsymbol{z}^1) q_{\boldsymbol{\phi}}(\boldsymbol{z}^1|
\boldsymbol{x}) && \textit{factors over layers} \\ 
\{z_i^l \perp z_j^l | \boldsymbol{z}^{l-1} \ &\cap \ 
z_i^1 \perp z_j^1 | \boldsymbol{x}\}_{\forall \ i \neq j,l} && \textit{factors over
random variables}
\end{aligned}
$$
<br>
</div>

<p class="body">

This is known as <a href="https://www.cs.cmu.edu/~epxing/Class/10708-17/notes-17/10708-scribe-lecture13.pdf" style="color:#FF7D33">mean field approximation</a> which 
yields a <b> tractable </b> surrogate distribution.

<h1 class="subhead"> 3.1.1 Making More Sense of the ELBO </h1>
<p class="body">

Understanding some variations of the ELBO really sheds light on the underlining intuitions of Variational Inference. For starters, we can rewrite the ELBO in the following form:

</p>

<div class="math">
<p class="formula">
<br>
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
<br>
</p>
</div>

<p class="body"> 
Note, the second expectation on the third line does not touch 

<span class="math">\(p_{\boldsymbol{\theta}}(\boldsymbol{x})\) </span>,

so it evaluates to 1. Furthermore, we should notice in the last line 
<a href = "https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence" style="color:#FF7D33"> KL Divergence</a> 

has made an appearance—which is a non-negative number that quantifies the distance between two distributions; where 0 is returned if the two distributions under concern are identical.
</p>

<figure>
<br>
  <img class="fig" src="/images/vae/kl-divergence.png" width="100%">
  <figcaption class="figcaption"> 
  Fig 2. Depiction of KL Divergence
  </figcaption>
<br>
</figure>

<p class = "body">
So, maximizing the lower bound is actually maximizing our original log probability and subtracting the distance between the intractable distribution and the surrogate distribution. Theoretically, we could set 

<span class = "math"> \(
q_{\boldsymbol{\phi}}(\boldsymbol{z|x})  = q_{\boldsymbol{\theta}}(\boldsymbol{z|x})
\)
</span> 

and our lower bound would be <b>tight</b>!

</p>

<pre>


</pre>
<h1 class="head"> 4. Relationship to Autoencoders </h1>

<p class = "body">
Thus far, all we have done is explore the variational component of VAEs by understanding 
the theoretical properties of the lower bound. Now, we will begin to understand the 
learning process of VAEs and their relationship to autoencoders.
</p>

<h1 class="subhead"> 4.1 Recognition Network </h1>
<p class="body">
We define the following:
</p>

<div class="formula">
<br>
$$
\begin{aligned}
\text{Recognition Network }:= \ & q_{\boldsymbol{\phi}}(\boldsymbol{z|x}) =
 q_{\boldsymbol{\phi}}(\boldsymbol{z}^2|
\boldsymbol{z}^1) q_{\boldsymbol{\phi}}(\boldsymbol{z}^1|
\boldsymbol{x}) \\
& q_{\boldsymbol{\phi}}
(\boldsymbol{z}^2|\boldsymbol{z}^1) \sim \mathcal{N}
(\boldsymbol{\mu}(\boldsymbol{z}^1,\boldsymbol{\phi}),
\boldsymbol{\Sigma}(\boldsymbol{z}^1,\boldsymbol{\phi})) \\
& q_{\boldsymbol{\phi}}
(\boldsymbol{z}^1|\boldsymbol{x}) \sim \mathcal{N}
(\boldsymbol{\mu}(\boldsymbol{x},\boldsymbol{\phi}),
\boldsymbol{\Sigma}(\boldsymbol{x},\boldsymbol{\phi})) \\
\end{aligned}
$$
<br>
</div>

<p class="body">
All of the distributions in the Recognition Network are Gaussians parameterized by
a mean vector,
<span class="math">

\(\boldsymbol{\mu}(\cdot)\)</span>, and a diagonal covariance matrix,
<span class="math">
\(\boldsymbol{\Sigma}(\cdot)\)</span>. Each of these are deterministic neural networks, e.g., FFNN, CNN or RNN parameterized by <span class="math">
\(\boldsymbol{\phi}\)</span>. The goal of the recognition network is to take in some observed data point, <span class="math">\(\boldsymbol{x}\)</span>, and learn the latent features, <span class="math">\(\boldsymbol{z}\)</span>, that best describe our data. In literature this is also known as the <b>encoder network</b> because we are encoding
information about <span class="math"> \(\boldsymbol{x}\)</span> into 
<span class="math">\(\boldsymbol{z}\)</span>.

</p>

<h1 class="subhead"> 4.2 Generative Network </h1>
<p class="body">
We define the following:
</p>

<div class="formula">
<br>
$$
\begin{aligned}
\text{Generative Network } := \ & p_{\boldsymbol{\theta}}(\boldsymbol{x,z}) = 
p(\boldsymbol{z}^2)p_{\boldsymbol{\theta}}(\boldsymbol{z}^1|\boldsymbol{z}^2)
p_{\boldsymbol{\theta}}(\boldsymbol{x}|\boldsymbol{z}^1) \\
& p(\boldsymbol{z}^2) \sim \mathcal{N}(\boldsymbol{0},\bf{I}) \\
&p_{\boldsymbol{\theta}}(\boldsymbol{z}^1|\boldsymbol{z}^2) \sim 
\mathcal{N}(\boldsymbol{\mu}(\boldsymbol{z}^2,\boldsymbol{\theta}),
\boldsymbol{\Sigma}(\boldsymbol{z}^2,\boldsymbol{\theta})) \\
& p_{\boldsymbol{\theta}}(\boldsymbol{x}|\boldsymbol{z}^1) \sim 
\mathcal{N}(\boldsymbol{\mu}(\boldsymbol{z}^1,\boldsymbol{\theta}),
\boldsymbol{\Sigma}(\boldsymbol{z}^1,\boldsymbol{\theta}))
\end{aligned}
$$
<br>
</div>

<p class="body">
We notice <span class="math"> 

\(p(\boldsymbol{z}^2)\)<span class="math"> is just the standard normal distribution and all other distributions are equivalent to the ones in the Recognition Network. The Generative Network uses the latent features to maximize the likelihood over the 
the joint distribution, 
<span class="math">
\(p_{\boldsymbol{\theta}}(\boldsymbol{x,z})\)</span>. After learning, we can sample from 
<span class="math"> \(\boldsymbol{x}_{new} \sim 
p_{\boldsymbol{\theta}}(\boldsymbol{x}_{observed}|\boldsymbol{z}^1)\)
</span> to generate a new example. Note,
<span class="math">
\(p_{\boldsymbol{\theta}}(\boldsymbol{x}|\boldsymbol{z}^1)\) =
\(p_{\boldsymbol{\theta}}(\boldsymbol{x}|\boldsymbol{z}^1,
\boldsymbol{z}^2)\) <span>, as defined by our graphical model. 
In literature this is known as the <b> decoder network </b> because we 
decode the information in <span class="math"> \(\boldsymbol{z}\)</span> to generate a new example, <span class="math"> \(\boldsymbol{x}_{new}\)<span class="math">. <br><br>
It should be noted the idea of a recognition and generative network is not new—where 
Helmholtz Machines, [@@helmholtzMachines], used an identical architecture but were trained using the Wake-sleep algorithm. 
</p>

<pre>

</pre>
<h1 class="head"> 5. The Learning Process </h1>
<p class="body">
Let us recall the objective function we are trying to maximize:

</p>

<div class="formula">
<br>
$$
\begin{aligned}
\mathcal{L}(\boldsymbol{x}) &= 
\mathbb{E}_{\boldsymbol{z}\sim q_{\boldsymbol{\phi}}(\boldsymbol{z|x})}
     \bigg[\log 
     \frac{p_{\boldsymbol{\theta}}(\boldsymbol{x},\boldsymbol{z})}
     {q_{\boldsymbol{\phi}}(\boldsymbol{z}|\boldsymbol{x})}
     \bigg] \\
&= \underbrace{
\mathbb{E}_{\boldsymbol{z}\sim q_{\boldsymbol{\phi}}(\boldsymbol{z|x})}
     [\log p_{\boldsymbol{\theta}}(\boldsymbol{x},\boldsymbol{z})]}_{\large \mathcal{L}_{recogn}(\boldsymbol{x})} - 
\underbrace{
\mathbb{E}_{\boldsymbol{z}\sim q_{\boldsymbol{\phi}}(\boldsymbol{z|x})}
[\log q_{\boldsymbol{\phi}}(\boldsymbol{z}|\boldsymbol{x})]}_{\large \mathcal{L}_{gen}
(\boldsymbol{x})}
\end{aligned}
$$
<br>
</div>

<p class="body">
We want to learn the parameters,
<span class="math">

\(\boldsymbol{\theta}\text{ and } \boldsymbol{\phi}\) </span>, in an end-to-end fashion. With this at hand, we can layout the computational graph:
</p>

<figure>
<br>
  <img class="fig" src="/images/vae/computational-graph1.png" width="80%">
  <figcaption class="figcaption"> 
  Fig 3. Computational graph of Variational Autoencoders
  </figcaption>
  <br>
</figure>

<p class="body">
I will acknowledge this may appear overwhelming but I did not fully understand the learning process nor the reparameterization trick until stepping through this graph in its entirety. A forward pass begins, bottom left, by taking an observed data point and sending it through the recognition network. Along the way, we sample a latent vector,
<span class="math"> 

\(\boldsymbol{z}^l\)</span>, send it through a deep network and generate the next set of parameters for an upstream distribution. After each latent layer has been sampled, we can forward propagate through the generative network. <br><br    >

Back propagation is actually fairly straightforward up until propagating through the latent nodes. Ultimately, the latent variables are <b>stochastic</b> and we cannot differentiate through a stochastic operator, red nodes. As a result, we would have to rely on gradient estimates via MCMC sampling which typically leads to high variance gradients and poor learning.
</p>

<h1 class="subhead">5.1 Reparameterization Trick</h1>
<p class="body">
Let us now define <span class="math"> 

\(\boldsymbol{z}^l\)</span> as a <b>deterministic</b> random variable:

<div class="formula">
<br>
$$
\begin{aligned}
\boldsymbol{z}^l &= g\bigg(\boldsymbol{\mu}(i,\boldsymbol{\phi}),
\boldsymbol{\Sigma}(i,\boldsymbol{\phi}),\boldsymbol{\epsilon}\bigg) \\
&=\boldsymbol{\mu}(i,\boldsymbol{\phi}) + 
\boldsymbol{\Sigma}(i,\boldsymbol{\phi})^{\frac{1}{2}}\otimes \boldsymbol{\epsilon}
&& \boldsymbol{\epsilon}\sim \mathcal{N}(0,\textbf{I})
\end{aligned}
$$
<br>
</p>

<p class="body">
<span class="math">

\(i\)<span> is just an argument and <span class="math">\(\boldsymbol{\Sigma}(i,\boldsymbol{\phi})^{\frac{1}{2}}\)</span> is the standard deviation as opposed to the variance. So, once we sample an <span class="math">\(\boldsymbol{\epsilon}\)</span>
from the standard normal we can use it to calculate <span class="math"> 
\(\boldsymbol{z}^l\)</span>. <b> Ultimately, the stochasticity now lies in 
<span class="math">\(\boldsymbol{\epsilon}\)</span> and not in <span class="math">
\(\boldsymbol{z}^l\)</span>.</b> Also, we can quickly show the reparameterized latent
variable still yields the same expected value and variance:

<div class="formula">
<br>
$$
\begin{aligned}
\mathbb{E}_{\boldsymbol{\epsilon}\sim\mathcal{N}(0,\textbf{I})}
[\boldsymbol{z}^l]&=
\boldsymbol{\mu} + 
\boldsymbol{\Sigma}^{\frac{1}{2}}
\mathbb{E}_{\boldsymbol{\epsilon}\sim\mathcal{N}(0,\textbf{I})}
[\boldsymbol{\epsilon}] \\
&=\boldsymbol{\mu} + 0 \\ \\
\text{Var}[\boldsymbol{z}^l]&= 
\mathbb{E}_{\boldsymbol{\epsilon}\sim\mathcal{N}(0,\textbf{I})}
\bigg[
\bigg(
\boldsymbol{\mu} + 
\boldsymbol{\Sigma}^{\frac{1}{2}}\boldsymbol{\epsilon}
\bigg)^2
\bigg] -  
\mathbb{E}_{\boldsymbol{\epsilon}\sim\mathcal{N}(0,\textbf{I})}
[\boldsymbol{z}^l]^2\\
&= \boldsymbol{\mu}^2 +
2\boldsymbol{\mu}\boldsymbol{\Sigma}^{\frac{1}{2}}
\mathbb{E}_{\boldsymbol{\epsilon}\sim\mathcal{N}(0,\textbf{I})}
[\boldsymbol{\epsilon}] + 
\boldsymbol{\Sigma}
\mathbb{E}_{\boldsymbol{\epsilon}\sim\mathcal{N}(0,\textbf{I})}
[\boldsymbol{\epsilon}^2] - \boldsymbol{\mu}^2 \\
&=\boldsymbol{\Sigma}\textbf{I}
\end{aligned}
$$
<br>
</div>

<p class="body">
where notation was simplified for brevity. <br> <br>

Now let us update the computational graph.
</p>

<figure>
  <img class="fig" src="/images/vae/vae-compgraph2.png" width="50%">
  <figcaption class="figcaption"> 
  Fig 4. Zoomed in computational graph of VAEs with the reparameterization trick applied
  </figcaption>
  <br>
</figure>

<p class="body">
We can now back propagate through all learnable parameters!
</p>

<pre>


</pre>
<h1 class="head"> Conclusion </h1>
<p class="body">
Now that we understand the theory of VAEs, we are in position to train a model and be informed
adjudicators of its performance. Luckily, <a class="hyperlinks" href="https://www.tensorflow.org/beta/tutorials/generative/cvae"> TensorFlow</a> and <a href="https://jmetzen.github.io/2015-11-27/vae.html" class="hyperlinks">
Jan Hendrik Metzen</a> have great tutorials on programmatic implementations. 
</p>




<script type="text/javascript">

if (!document.getElementById('mathjaxscript_pelican_#%@#$@#')) {
    var align = "center",
        indent = "0em",
        linebreak = "false";

    if (false) {
        align = (screen.width < 768) ? "left" : align;
        indent = (screen.width < 768) ? "0em" : indent;
        linebreak = (screen.width < 768) ? 'true' : linebreak;
    }

    var mathjaxscript = document.createElement('script');
    mathjaxscript.id = 'mathjaxscript_pelican_#%@#$@#';
    mathjaxscript.type = 'text/javascript';
    mathjaxscript.src = 'https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.3/latest.js?config=TeX-AMS-MML_HTMLorMML';

    var configscript = document.createElement('script');
    configscript.type = 'text/x-mathjax-config';
    configscript[(window.opera ? "innerHTML" : "text")] =
        "MathJax.Hub.Config({" +
        "    config: ['MMLorHTML.js']," +
        "    TeX: { extensions: ['AMSmath.js','AMSsymbols.js','noErrors.js','noUndefined.js'], equationNumbers: { autoNumber: 'none' } }," +
        "    jax: ['input/TeX','input/MathML','output/HTML-CSS']," +
        "    extensions: ['tex2jax.js','mml2jax.js','MathMenu.js','MathZoom.js']," +
        "    displayAlign: '"+ align +"'," +
        "    displayIndent: '"+ indent +"'," +
        "    showMathMenu: true," +
        "    messageStyle: 'normal'," +
        "    tex2jax: { " +
        "        inlineMath: [ ['\\\\(','\\\\)'] ], " +
        "        displayMath: [ ['$$','$$'] ]," +
        "        processEscapes: true," +
        "        preview: 'TeX'," +
        "    }, " +
        "    'HTML-CSS': { " +
        "        fonts: ['STIX', 'TeX']," +
        "        styles: { '.MathJax_Display, .MathJax .mo, .MathJax .mi, .MathJax .mn': {color: 'inherit ! important'} }," +
        "        linebreaks: { automatic: "+ linebreak +", width: '90% container' }," +
        "    }, " +
        "}); " +
        "if ('default' !== 'default') {" +
            "MathJax.Hub.Register.StartupHook('HTML-CSS Jax Ready',function () {" +
                "var VARIANT = MathJax.OutputJax['HTML-CSS'].FONTDATA.VARIANT;" +
                "VARIANT['normal'].fonts.unshift('MathJax_default');" +
                "VARIANT['bold'].fonts.unshift('MathJax_default-bold');" +
                "VARIANT['italic'].fonts.unshift('MathJax_default-italic');" +
                "VARIANT['-tex-mathit'].fonts.unshift('MathJax_default-italic');" +
            "});" +
            "MathJax.Hub.Register.StartupHook('SVG Jax Ready',function () {" +
                "var VARIANT = MathJax.OutputJax.SVG.FONTDATA.VARIANT;" +
                "VARIANT['normal'].fonts.unshift('MathJax_default');" +
                "VARIANT['bold'].fonts.unshift('MathJax_default-bold');" +
                "VARIANT['italic'].fonts.unshift('MathJax_default-italic');" +
                "VARIANT['-tex-mathit'].fonts.unshift('MathJax_default-italic');" +
            "});" +
        "}";

    (document.body || document.getElementsByTagName('head')[0]).appendChild(configscript);
    (document.body || document.getElementsByTagName('head')[0]).appendChild(mathjaxscript);
}

</script>
