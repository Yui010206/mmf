(window.webpackJsonp=window.webpackJsonp||[]).push([[12],{114:function(e,t,n){"use strict";n.r(t),n.d(t,"frontMatter",(function(){return o})),n.d(t,"metadata",(function(){return i})),n.d(t,"rightToc",(function(){return c})),n.d(t,"default",(function(){return p}));var a=n(2),l=n(6),r=(n(0),n(124)),o={id:"other_challenges",title:"Other Challenges",sidebar_label:"Other Challenges"},i={id:"challenges/other_challenges",title:"Other Challenges",description:"Challenge Participation",source:"@site/docs/challenges/other_challenges.md",permalink:"/docs/challenges/other_challenges",editUrl:"https://github.com/facebookresearch/mmf/edit/master/website/docs/challenges/other_challenges.md",lastUpdatedBy:"Amanpreet Singh",lastUpdatedAt:1591757356,sidebar_label:"Other Challenges",sidebar:"docs",previous:{title:"Hateful Memes Challenge",permalink:"/docs/challenges/hateful_memes_challenge"}},c=[{value:"TextVQA challenge",id:"textvqa-challenge",children:[]},{value:"VQA Challenge",id:"vqa-challenge",children:[]}],s={rightToc:c};function p(e){var t=e.components,n=Object(l.a)(e,["components"]);return Object(r.b)("wrapper",Object(a.a)({},s,n,{components:t,mdxType:"MDXLayout"}),Object(r.b)("h1",{id:"challenge-participation"},"Challenge Participation"),Object(r.b)("p",null,Object(r.b)("strong",{parentName:"p"},"[Outdated]")," A new version of this will be uploaded soon"),Object(r.b)("p",null,"Participating in EvalAI challenges is really easy using MMF. We will show how to\ndo inference for two challenges here:"),Object(r.b)("pre",null,Object(r.b)("code",Object(a.a)({parentName:"pre"},{className:"language-eval_rst"}),".. note::\n\n  This section assumes that you have downloaded data following the Quickstart_ tutorial.\n\n.. _Quickstart: ./quickstart\n")),Object(r.b)("h2",{id:"textvqa-challenge"},"TextVQA challenge"),Object(r.b)("p",null,"TextVQA challenge is available at ",Object(r.b)("a",Object(a.a)({parentName:"p"},{href:"https://evalai.cloudcv.org/web/challenges/challenge-page/244/overview"}),"this link"),".\nCurrently, LoRRA is the SoTA on TextVQA. To do inference on val set using LoRRA, follow the steps below:"),Object(r.b)("pre",null,Object(r.b)("code",Object(a.a)({parentName:"pre"},{}),"# Download the model first\ncd ~/mmf/data\nmkdir -p models && cd models;\n# Get link from the table above and extract if needed\nwget https://dl.fbaipublicfiles.com/pythia/pretrained_models/textvqa/lorra_best.pth\n\ncd ../..\n# Replace datasets and model with corresponding key for other pretrained models\npython tools/run.py --datasets textvqa --model lorra --config configs/vqa/textvqa/lorra.yaml \\\n--run_type val --evalai_inference 1 --resume_file data/models/lorra_best.pth\n")),Object(r.b)("p",null,"In the printed log, MMF will mention where it wrote the JSON file it created.\nUpload that file on EvalAI:"),Object(r.b)("pre",null,Object(r.b)("code",Object(a.a)({parentName:"pre"},{}),"> Go to https://evalai.cloudcv.org/web/challenges/challenge-page/244/overview\n> Select Submit Tab\n> Select Validation Phase\n> Select the file by click Upload file\n> Write a model name\n> Upload\n")),Object(r.b)("p",null,"To check your results, go in 'My submissions' section and select 'Validation Phase' and click on 'Result file'."),Object(r.b)("p",null,"Now, you can either edit the LoRRA model to create your own model on top of it or create your own model inside MMF to\nbeat LoRRA in challenge."),Object(r.b)("h2",{id:"vqa-challenge"},"VQA Challenge"),Object(r.b)("p",null,"Similar to TextVQA challenge, VQA Challenge is available at ",Object(r.b)("a",Object(a.a)({parentName:"p"},{href:"https://evalai.cloudcv.org/web/challenges/challenge-page/163/overview"}),"this link"),". You can either select Pythia as your base model\nor LoRRA model (available soon for VQA2) from the table in ",Object(r.b)("a",Object(a.a)({parentName:"p"},{href:"pretrained_models"}),"pretrained models")," section as a base."),Object(r.b)("p",null,"Follow the same steps above, replacing ",Object(r.b)("inlineCode",{parentName:"p"},"--model")," with ",Object(r.b)("inlineCode",{parentName:"p"},"pythia")," or ",Object(r.b)("inlineCode",{parentName:"p"},"lorra")," and ",Object(r.b)("inlineCode",{parentName:"p"},"--datasets")," with ",Object(r.b)("inlineCode",{parentName:"p"},"vqa2"),".\nAlso, replace the config accordingly. Here are example commands for using Pythia to do inference on test set of VQA2."),Object(r.b)("pre",null,Object(r.b)("code",Object(a.a)({parentName:"pre"},{}),"# Download the model first\ncd ~/mmf/data\nmkdir -p models && cd models;\n# Get link from the table above and extract if needed\nwget https://dl.fbaipublicfiles.com/pythia/pretrained_models/textvqa/pythia_train_val.pth\n\ncd ../..\n# Replace datasets and model with corresponding key for other pretrained models\npython tools/run.py --datasets vqa2 --model pythia --config configs/vqa/vqa2/pythia.yaml \\\n--run_type inference --evalai_inference 1 --resume_file data/models/pythia_train_val.pth\n")),Object(r.b)("p",null,"Now, similar to TextVQA challenge follow the steps to upload the prediction file, but this time to ",Object(r.b)("inlineCode",{parentName:"p"},"test-dev")," phase."))}p.isMDXComponent=!0},124:function(e,t,n){"use strict";n.d(t,"a",(function(){return d})),n.d(t,"b",(function(){return u}));var a=n(0),l=n.n(a);function r(e,t,n){return t in e?Object.defineProperty(e,t,{value:n,enumerable:!0,configurable:!0,writable:!0}):e[t]=n,e}function o(e,t){var n=Object.keys(e);if(Object.getOwnPropertySymbols){var a=Object.getOwnPropertySymbols(e);t&&(a=a.filter((function(t){return Object.getOwnPropertyDescriptor(e,t).enumerable}))),n.push.apply(n,a)}return n}function i(e){for(var t=1;t<arguments.length;t++){var n=null!=arguments[t]?arguments[t]:{};t%2?o(Object(n),!0).forEach((function(t){r(e,t,n[t])})):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(n)):o(Object(n)).forEach((function(t){Object.defineProperty(e,t,Object.getOwnPropertyDescriptor(n,t))}))}return e}function c(e,t){if(null==e)return{};var n,a,l=function(e,t){if(null==e)return{};var n,a,l={},r=Object.keys(e);for(a=0;a<r.length;a++)n=r[a],t.indexOf(n)>=0||(l[n]=e[n]);return l}(e,t);if(Object.getOwnPropertySymbols){var r=Object.getOwnPropertySymbols(e);for(a=0;a<r.length;a++)n=r[a],t.indexOf(n)>=0||Object.prototype.propertyIsEnumerable.call(e,n)&&(l[n]=e[n])}return l}var s=l.a.createContext({}),p=function(e){var t=l.a.useContext(s),n=t;return e&&(n="function"==typeof e?e(t):i(i({},t),e)),n},d=function(e){var t=p(e.components);return l.a.createElement(s.Provider,{value:t},e.children)},b={inlineCode:"code",wrapper:function(e){var t=e.children;return l.a.createElement(l.a.Fragment,{},t)}},h=l.a.forwardRef((function(e,t){var n=e.components,a=e.mdxType,r=e.originalType,o=e.parentName,s=c(e,["components","mdxType","originalType","parentName"]),d=p(n),h=a,u=d["".concat(o,".").concat(h)]||d[h]||b[h]||r;return n?l.a.createElement(u,i(i({ref:t},s),{},{components:n})):l.a.createElement(u,i({ref:t},s))}));function u(e,t){var n=arguments,a=t&&t.mdxType;if("string"==typeof e||a){var r=n.length,o=new Array(r);o[0]=h;var i={};for(var c in t)hasOwnProperty.call(t,c)&&(i[c]=t[c]);i.originalType=e,i.mdxType="string"==typeof e?e:a,o[1]=i;for(var s=2;s<r;s++)o[s]=n[s];return l.a.createElement.apply(null,o)}return l.a.createElement.apply(null,n)}h.displayName="MDXCreateElement"}}]);