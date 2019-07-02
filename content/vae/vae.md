---
Title: Variational Autoencoders
Date: 01/01/2019
Tags: pgm, generative_models 
Category: pgm
Slug: test-post
Summary: SOME TEXT
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
which are essentially the underlining features that best explain our observed data
<span class = "math">\(\boldsymbol{x}\)</span>.
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
Furthermore, it is challenging to learn the parameters of both distributions in an end-to-end fashion. [@@wakeSleep] developed the Wake-sleep algorithm which was able to learn all parameters to this exact problem via two objective functions and back-propagation. While this algorithm <i> masked </i> the problem at hand, even Geoffrey Hinton 
<a href="https://www.youtube.com/watch?v=VKpc_z7b9I0" style="color:#FF7D33"> 
acknowledged</a>, and I quote, <i> "the basic idea behind these variational methods sounds crazy." </i> Luckily, [@@vae] offered a better solution using the <i>
reparameterization trick </i> and deep generative models have really been in resurgence since! 

</p>

<h1 class="head"> 2. Problem Scenario </h1>

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
