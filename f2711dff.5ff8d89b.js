(window.webpackJsonp=window.webpackJsonp||[]).push([[17],{120:function(e,t,n){"use strict";n.r(t),n.d(t,"frontMatter",(function(){return i})),n.d(t,"metadata",(function(){return l})),n.d(t,"rightToc",(function(){return s})),n.d(t,"default",(function(){return u}));var r=n(2),a=n(6),o=(n(0),n(124)),i={id:"installation",title:"Installation",sidebar_label:"Installation"},l={id:"getting_started/installation",title:"Installation",description:"MMF is tested on Python 3.7+ and PyTorch 1.5.",source:"@site/docs/getting_started/installation.md",permalink:"/docs/getting_started/installation",editUrl:"https://github.com/facebookresearch/mmf/edit/master/website/docs/getting_started/installation.md",lastUpdatedBy:"Amanpreet Singh",lastUpdatedAt:1591757356,sidebar_label:"Installation",sidebar:"docs",next:{title:"MMF Features",permalink:"/docs/getting_started/features"}},s=[{value:"Install using pip",id:"install-using-pip",children:[]},{value:"Install from source",id:"install-from-source",children:[]},{value:"Running tests",id:"running-tests",children:[]}],c={rightToc:s};function u(e){var t=e.components,n=Object(a.a)(e,["components"]);return Object(o.b)("wrapper",Object(r.a)({},c,n,{components:t,mdxType:"MDXLayout"}),Object(o.b)("p",null,"MMF is tested on Python 3.7+ and PyTorch 1.5."),Object(o.b)("h2",{id:"install-using-pip"},"Install using pip"),Object(o.b)("p",null,"MMF can be installed from pip with following command:"),Object(o.b)("pre",null,Object(o.b)("code",Object(r.a)({parentName:"pre"},{className:"language-bash"}),"pip install --upgrade --pre mmf\n")),Object(o.b)("p",null,"Use this if:"),Object(o.b)("ul",null,Object(o.b)("li",{parentName:"ul"},"You are using MMF as a library and not developing inside MMF. Have a look at extending MMF tutorial."),Object(o.b)("li",{parentName:"ul"},"You want easy installation and don't care about up-to-date features. Note that, pip packages are always outdated compared to installing from source")),Object(o.b)("h2",{id:"install-from-source"},"Install from source"),Object(o.b)("p",null,"To install from source, do:"),Object(o.b)("pre",null,Object(o.b)("code",Object(r.a)({parentName:"pre"},{className:"language-bash"}),"git clone https://github.com/facebookresearch/mmf.git\ncd mmf\npip install --editable .\n")),Object(o.b)("h2",{id:"running-tests"},"Running tests"),Object(o.b)("p",null,"MMF uses pytest for testing purposes. To ensure everything and run tests at your end do:"),Object(o.b)("pre",null,Object(o.b)("code",Object(r.a)({parentName:"pre"},{className:"language-bash"}),"pytest ./tests/\n")))}u.isMDXComponent=!0},124:function(e,t,n){"use strict";n.d(t,"a",(function(){return p})),n.d(t,"b",(function(){return f}));var r=n(0),a=n.n(r);function o(e,t,n){return t in e?Object.defineProperty(e,t,{value:n,enumerable:!0,configurable:!0,writable:!0}):e[t]=n,e}function i(e,t){var n=Object.keys(e);if(Object.getOwnPropertySymbols){var r=Object.getOwnPropertySymbols(e);t&&(r=r.filter((function(t){return Object.getOwnPropertyDescriptor(e,t).enumerable}))),n.push.apply(n,r)}return n}function l(e){for(var t=1;t<arguments.length;t++){var n=null!=arguments[t]?arguments[t]:{};t%2?i(Object(n),!0).forEach((function(t){o(e,t,n[t])})):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(n)):i(Object(n)).forEach((function(t){Object.defineProperty(e,t,Object.getOwnPropertyDescriptor(n,t))}))}return e}function s(e,t){if(null==e)return{};var n,r,a=function(e,t){if(null==e)return{};var n,r,a={},o=Object.keys(e);for(r=0;r<o.length;r++)n=o[r],t.indexOf(n)>=0||(a[n]=e[n]);return a}(e,t);if(Object.getOwnPropertySymbols){var o=Object.getOwnPropertySymbols(e);for(r=0;r<o.length;r++)n=o[r],t.indexOf(n)>=0||Object.prototype.propertyIsEnumerable.call(e,n)&&(a[n]=e[n])}return a}var c=a.a.createContext({}),u=function(e){var t=a.a.useContext(c),n=t;return e&&(n="function"==typeof e?e(t):l(l({},t),e)),n},p=function(e){var t=u(e.components);return a.a.createElement(c.Provider,{value:t},e.children)},b={inlineCode:"code",wrapper:function(e){var t=e.children;return a.a.createElement(a.a.Fragment,{},t)}},d=a.a.forwardRef((function(e,t){var n=e.components,r=e.mdxType,o=e.originalType,i=e.parentName,c=s(e,["components","mdxType","originalType","parentName"]),p=u(n),d=r,f=p["".concat(i,".").concat(d)]||p[d]||b[d]||o;return n?a.a.createElement(f,l(l({ref:t},c),{},{components:n})):a.a.createElement(f,l({ref:t},c))}));function f(e,t){var n=arguments,r=t&&t.mdxType;if("string"==typeof e||r){var o=n.length,i=new Array(o);i[0]=d;var l={};for(var s in t)hasOwnProperty.call(t,s)&&(l[s]=t[s]);l.originalType=e,l.mdxType="string"==typeof e?e:r,i[1]=l;for(var c=2;c<o;c++)i[c]=n[c];return a.a.createElement.apply(null,i)}return a.a.createElement.apply(null,n)}d.displayName="MDXCreateElement"}}]);