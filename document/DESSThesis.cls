%
% LLNCS DOCUMENT CLASS -- version 2.10
% for LaTeX2e
%
\NeedsTeXFormat{LaTeX2e}[1995/12/01]
\ProvidesClass{DESSThesis}[2012/02/15]

\typeout{***********************************************************************}
\typeout{* Data Science in the Economic and Social Sciences *}
\typeout{***********************************************************************}

\LoadClass[11pt,a4paper,twoside,titlepage,headings=big,chapterprefix=true]{scrbook}\RequirePackage[twoside,bindingoffset=0.5cm,includeheadfoot,twoside,top=2cm,bottom=2cm,inner=3cm, outer=2.5cm]{geometry} %Page layout

\RequirePackage[T1]{fontenc}
\RequirePackage[utf8]{inputenc}
\RequirePackage[english]{babel}
\RequirePackage[babel]{csquotes} % requires texlive-latex-extra
\RequirePackage[style=numeric, backend=biber]{biblatex}
\RequirePackage[colorlinks,hidelinks, pdfpagelabels,pdfstartview = FitH, bookmarksopen = true,bookmarksnumbered = true, citecolor=black, linkcolor = black, urlcolor=black, plainpages = false, hypertexnames = false]{hyperref} %links within the doc
\usepackage[dvipsnames]{xcolor}

%layout packages
\RequirePackage{lmodern} %Change font size of chapter titles
\RequirePackage{fancyhdr}
\pagestyle{fancy}
\fancyhead{} % clear all header fields
\fancyhead[RO]{\textsl{\rightmark}}
\fancyhead[LE]{\textsl{\leftmark}}
\fancyfoot{} % clear all footer fields
\fancyfoot[CE,CO]{\thepage}
\fancyfoot[LO,RO]{}
\fancyfoot[LE,RE]{}
\renewcommand{\headrulewidth}{.4pt}
\renewcommand{\footrulewidth}{.4pt}
\RequirePackage[textfont=it]{caption} %add labelfont when other style preferred
\setcapindent{0pt}
% \setlength{\parindent}{0cm}

% \captionsetup{indention= -1.95cm} %second line of caption is aligned with leftmost side
%\RequirePackage[font=small,margin=2.5cm,labelfont=bf]{caption}
% \RequirePackage{layout} %print layout options 


%Necessary Packages
\RequirePackage{mathptmx} %Sets the font of the entire document + math mode to times new roman
\DeclareMathAlphabet{\mathcal}{OMS}{cmsy}{m}{n} %Resets the mathcal font to default
% \RequirePackage{ae}
\RequirePackage{amsmath}
\RequirePackage{amsxtra}
\RequirePackage{amsfonts}
\RequirePackage{amssymb}
\RequirePackage{amsthm}
\RequirePackage{blkarray} % Blockwise defined arrays % requires texlive-latex-extra


% \RequirePackage{breakurl}
% \RequirePackage{dsfont} %Requires installation of doublestroke package
% \RequirePackage{enumerate}
\RequirePackage{enumitem} % requires texlive-latex-extra
% \RequirePackage{eurosym}
% \RequirePackage{epsfig}
\RequirePackage{float}
\RequirePackage{graphicx} %Include external graphics
\RequirePackage{mathtools} %eg coloneqq defined here
\RequirePackage{multicol}
\RequirePackage{setspace} %change spacing between lines
\RequirePackage{subfigure} %Necessary package for numerating the figures
\RequirePackage{tabularx} %tables with linebreak
\RequirePackage{textcomp}
\RequirePackage{tikz} %Drawing library
\RequirePackage{pgfplots}
\usetikzlibrary{arrows, backgrounds, calc, datavisualization, datavisualization.formats.functions,%
				decorations, graphs, graphs.standard, positioning, shapes, shadows}
\RequirePackage{times}
% \RequirePackage{verbatim}
% \RequirePackage{wasysym}
% \RequirePackage{xkeyval}
% \RequirePackage{xargs}  

\rmfamily

\widowpenalty=10000
\clubpenalty=10000

\AtBeginDocument{

\addtokomafont{chapterprefix}{\raggedleft}
\addtokomafont{section}{\huge}
\addtokomafont{subsection}{\Large}
\addtokomafont{subsubsection}{\large}

\renewcommand*{\chapterformat}{%
	\mbox{\scalebox{1.5}{{\nobreakspace}}%
		\scalebox{4}{\color{gray}\thechapter\autodot}\enskip}
	{\par%
		\kern-.75\ht\strutbox\rule{\linewidth}{.8pt}%
	}%	
}
\pagenumbering{Roman} %start with roman literals for counting
}


\renewenvironment{figure}
               {\setcounter{subfigure}{0}
                \@float{figure}}
               {\end@float}
\renewenvironment{table}
               {\setcounter{subtable}{0}
                \@float{table}}
               {\end@float}



\newenvironment{abstract}
{\noindent{\normalfont\bfseries\large\abstractname}
	%\vskip10pt%
	%\noindent
	\normalfont\itshape}%
{\par}

\renewcommand{\baselinestretch}{1.2} %Spacing between lines

\renewcommand{\labelitemi}{--}


	
\DeclareGraphicsRule{.pdf}{pdf}{.pdf}{}
