(window.webpackJsonp=window.webpackJsonp||[]).push([[11],{113:function(e,t,n){"use strict";n.r(t),n.d(t,"frontMatter",(function(){return l})),n.d(t,"metadata",(function(){return o})),n.d(t,"rightToc",(function(){return c})),n.d(t,"default",(function(){return u}));var a=n(2),r=n(6),i=(n(0),n(124)),l={id:"hateful_memes_challenge",title:"Hateful Memes Challenge",sidebar_label:"Hateful Memes Challenge"},o={id:"challenges/hateful_memes_challenge",title:"Hateful Memes Challenge",description:"The Hateful Memes challenge is available at this link.",source:"@site/docs/challenges/hateful_memes_challenge.md",permalink:"/docs/challenges/hateful_memes_challenge",editUrl:"https://github.com/facebookresearch/mmf/edit/master/website/docs/challenges/hateful_memes_challenge.md",lastUpdatedBy:"Amanpreet Singh",lastUpdatedAt:1591757356,sidebar_label:"Hateful Memes Challenge",sidebar:"docs",previous:{title:"Tutorial: Late Fusion",permalink:"/docs/tutorials/late_fusion"},next:{title:"Other Challenges",permalink:"/docs/challenges/other_challenges"}},c=[{value:"Installation and Preparing the dataset",id:"installation-and-preparing-the-dataset",children:[]},{value:"Training and Evaluation",id:"training-and-evaluation",children:[{value:"Training",id:"training",children:[]},{value:"Evaluation",id:"evaluation",children:[]}]},{value:"Predictions for Challenge",id:"predictions-for-challenge",children:[]},{value:"Submission for Challenge",id:"submission-for-challenge",children:[]},{value:"Building on top of MMF and Open Sourcing your code",id:"building-on-top-of-mmf-and-open-sourcing-your-code",children:[]}],s={rightToc:c};function u(e){var t=e.components,n=Object(r.a)(e,["components"]);return Object(i.b)("wrapper",Object(a.a)({},s,n,{components:t,mdxType:"MDXLayout"}),Object(i.b)("p",null,"The Hateful Memes challenge is available at ",Object(i.b)("a",Object(a.a)({parentName:"p"},{href:"https://www.drivendata.org/competitions/64/hateful-memes"}),"this link"),"."),Object(i.b)("p",null,"In MMF, we provide the starter code and baseline pretrained models for this challenge and the configurations used for training the reported baselines. For more details check ",Object(i.b)("a",Object(a.a)({parentName:"p"},{href:"https://github.com/facebookresearch/mmf/tree/master/projects/hateful_memes"}),"this link"),"."),Object(i.b)("p",null,"In this tutorial, we provide steps for running training and evaluation with MMBT model on hateful memes dataset and generating submission file for the challenge. The same steps can be used for your own models."),Object(i.b)("h2",{id:"installation-and-preparing-the-dataset"},"Installation and Preparing the dataset"),Object(i.b)("p",null,"Follow the prerequisites for installation and dataset ",Object(i.b)("a",Object(a.a)({parentName:"p"},{href:"https://github.com/facebookresearch/mmf/tree/master/projects/hateful_memes#prerequisites"}),"here"),"."),Object(i.b)("h2",{id:"training-and-evaluation"},"Training and Evaluation"),Object(i.b)("h3",{id:"training"},"Training"),Object(i.b)("p",null,"For running training on train set, run the following command:"),Object(i.b)("pre",null,Object(i.b)("code",Object(a.a)({parentName:"pre"},{}),"mmf_run config=projects/hateful_memes/configs/mmbt/defaults.yaml model=mmbt dataset=hateful_memes training.run_type=train_val\n")),Object(i.b)("p",null,"This will train the ",Object(i.b)("inlineCode",{parentName:"p"},"mmbt")," model on the dataset and generate the checkpoints and best trained model (",Object(i.b)("inlineCode",{parentName:"p"},"mmbt_final.pth"),") will be stored in the ",Object(i.b)("inlineCode",{parentName:"p"},"./save")," directory by default."),Object(i.b)("h3",{id:"evaluation"},"Evaluation"),Object(i.b)("p",null,"Next run evaluation on the validation set:"),Object(i.b)("pre",null,Object(i.b)("code",Object(a.a)({parentName:"pre"},{}),"mmf_run config=projects/hateful_memes/configs/mmbt/defaults.yaml \\\n    model=mmbt \\\n    dataset=hateful_memes \\\n    training.run_type=val \\\n    resume_file=./save/mmbt_final.pth\n")),Object(i.b)("p",null,"This will give you the performance of your model on the validation set. The metrics are AUROC, ACC, Binary F1 etc."),Object(i.b)("h2",{id:"predictions-for-challenge"},"Predictions for Challenge"),Object(i.b)("p",null,"After we trained the model and evaluated on the validation set, we will generate the predictions on the test set. The prediction file should contain the following three columns:"),Object(i.b)("ul",null,Object(i.b)("li",{parentName:"ul"},"Meme identification number, ",Object(i.b)("inlineCode",{parentName:"li"},"id")),Object(i.b)("li",{parentName:"ul"},"Probability that the meme is hateful, ",Object(i.b)("inlineCode",{parentName:"li"},"proba")),Object(i.b)("li",{parentName:"ul"},"Binary label that the meme is hateful (1) or non-hateful (0), ",Object(i.b)("inlineCode",{parentName:"li"},"label"))),Object(i.b)("p",null,"With MMF you can directly generate the predictions in the required submission format with the following command:"),Object(i.b)("pre",null,Object(i.b)("code",Object(a.a)({parentName:"pre"},{}),"mmf_predict config=projects/hateful_memes/configs/mmbt/defaults.yaml \\\n    model=mmbt \\\n    dataset=hateful_memes \\\n    run_type=test\n")),Object(i.b)("p",null,"This command will output where the generated predictions csv file is stored."),Object(i.b)("h2",{id:"submission-for-challenge"},"Submission for Challenge"),Object(i.b)("p",null,"Next you can upload the generated csv file on DrivenData in their ",Object(i.b)("a",Object(a.a)({parentName:"p"},{href:"https://www.drivendata.org/competitions/64/hateful-memes/submissions/"}),"submissions")," page for Hateful Memes."),Object(i.b)("p",null,"More details will be added once the challenge submission phase is live."),Object(i.b)("h2",{id:"building-on-top-of-mmf-and-open-sourcing-your-code"},"Building on top of MMF and Open Sourcing your code"),Object(i.b)("p",null,"To understand how you build on top of MMF for your own custom models and then open source your code, take a look at this ",Object(i.b)("a",Object(a.a)({parentName:"p"},{href:"https://github.com/apsdehal/hm_example_mmf"}),"example"),"."))}u.isMDXComponent=!0},124:function(e,t,n){"use strict";n.d(t,"a",(function(){return m})),n.d(t,"b",(function(){return p}));var a=n(0),r=n.n(a);function i(e,t,n){return t in e?Object.defineProperty(e,t,{value:n,enumerable:!0,configurable:!0,writable:!0}):e[t]=n,e}function l(e,t){var n=Object.keys(e);if(Object.getOwnPropertySymbols){var a=Object.getOwnPropertySymbols(e);t&&(a=a.filter((function(t){return Object.getOwnPropertyDescriptor(e,t).enumerable}))),n.push.apply(n,a)}return n}function o(e){for(var t=1;t<arguments.length;t++){var n=null!=arguments[t]?arguments[t]:{};t%2?l(Object(n),!0).forEach((function(t){i(e,t,n[t])})):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(n)):l(Object(n)).forEach((function(t){Object.defineProperty(e,t,Object.getOwnPropertyDescriptor(n,t))}))}return e}function c(e,t){if(null==e)return{};var n,a,r=function(e,t){if(null==e)return{};var n,a,r={},i=Object.keys(e);for(a=0;a<i.length;a++)n=i[a],t.indexOf(n)>=0||(r[n]=e[n]);return r}(e,t);if(Object.getOwnPropertySymbols){var i=Object.getOwnPropertySymbols(e);for(a=0;a<i.length;a++)n=i[a],t.indexOf(n)>=0||Object.prototype.propertyIsEnumerable.call(e,n)&&(r[n]=e[n])}return r}var s=r.a.createContext({}),u=function(e){var t=r.a.useContext(s),n=t;return e&&(n="function"==typeof e?e(t):o(o({},t),e)),n},m=function(e){var t=u(e.components);return r.a.createElement(s.Provider,{value:t},e.children)},d={inlineCode:"code",wrapper:function(e){var t=e.children;return r.a.createElement(r.a.Fragment,{},t)}},b=r.a.forwardRef((function(e,t){var n=e.components,a=e.mdxType,i=e.originalType,l=e.parentName,s=c(e,["components","mdxType","originalType","parentName"]),m=u(n),b=a,p=m["".concat(l,".").concat(b)]||m[b]||d[b]||i;return n?r.a.createElement(p,o(o({ref:t},s),{},{components:n})):r.a.createElement(p,o({ref:t},s))}));function p(e,t){var n=arguments,a=t&&t.mdxType;if("string"==typeof e||a){var i=n.length,l=new Array(i);l[0]=b;var o={};for(var c in t)hasOwnProperty.call(t,c)&&(o[c]=t[c]);o.originalType=e,o.mdxType="string"==typeof e?e:a,l[1]=o;for(var s=2;s<i;s++)l[s]=n[s];return r.a.createElement.apply(null,l)}return r.a.createElement.apply(null,n)}b.displayName="MDXCreateElement"}}]);