#!/usr/bin/env wolframscript
(* ::Package:: *)

(* ::Section:: *)
(*Add rotations*)


(* ::Text:: *)
(*Get skew matrix*)


skew[vec_]:=Module[{b},
  b=Table[bb[i],{i,3}];
  Normal[CoefficientArrays[Cross[vec,b],b][[2]]]
]


(* ::Text:: *)
(*Rotation matrix from quaternions (Krenk (3.52))*)


R[{q0_,q_}]:=(q0^2-Dot[q,q])IdentityMatrix[3]+2q0 skew[q]+2q\[TensorProduct]q;


(* ::Text:: *)
(*Quaternion addition, first q then p (Krenk(3.63))*)


add[{q0_,q_},{p0_,p_}]:={
  p0 q0-p . q,
  p0 q+q0 p+p\[Cross]q
}


(* ::Text:: *)
(*Apply a rotation to a vector*)


R[{q0,{q1,q2,q3}}] . {r1,r2,r3}


(* ::Text:: *)
(*Add two rotations*)


add[{q0,{q1,q2,q3}},{p0,{p1,p2,p3}}]


(* ::Section:: *)
(*Integration of angular velocity*)


(* ::Text:: *)
(*Since quaternions are not part of a vector space, we can not directly integrate them. Instead we need to solve a differential equation. A good overview over this is given in:*)
(*	Boyle, M. 2017. The integration of angular velocity. Advances in Applied Clifford Algebras. 27, 3 (2017), 2345\[Dash]2374. DOI:https://doi.org/10.1007/s00006-017-0793-z*)
(*The right hand side of the differential equation reads (where, q is the current quaternion and \[Omega] is the angular velocity)*)


angularVelocityRHS[q_,\[Omega]_]:=add[q,{0,\[Omega]}]/2


(* ::Text:: *)
(*Integrate angular velocity (the initial triad is always the identity triad)*)


integrateAngularVelocity[angularVel_]:=NDSolve[
	Join[
		Thread[{q0'[t],q1'[t],q2'[t],q3'[t]}==Flatten[angularVelocityRHS[{q0[t],{q1[t],q2[t],q3[t]}},angularVel[t]]]],
		{q0[0]==1,q1[0]==0,q2[0]==0,q3[0]==0}
	],
	{q0,q1,q2,q3},
	{t,tStart,tEnd}, AccuracyGoal->20,PrecisionGoal->20,WorkingPrecision->35
]


(* ::Text:: *)
(*Apply the function to an example*)


angularVel[t_]:={2,0,5-2t};
tStart=0;
tEnd=1;
sol=integrateAngularVelocity[angularVel];
Plot[Evaluate[{q0[t],q1[t],q2[t],q3[t]}/.sol],{t,tStart,tEnd},PlotStyle->Automatic]
Plot[Evaluate[Norm[{q0[t],q1[t],q2[t],q3[t]}]/.sol],{t,tStart,tEnd},PlotStyle->Automatic]
{q0[t], q1[t], q2[t],q3[t]}/.sol/.t->tEnd


(* ::Subsection:: *)
(*Integrate curvature of parametric curve*)


(* ::Text:: *)
(*Integrate angular velocity for a parametric curve. The angular velocity is equal to the curvature vector*)


curvature[r_,t_]:=(Cross[D[r,t],D[r,t,t]])/(D[r,t] . D[r,t])


(* ::Text:: *)
(*Get the triad along a scaled helix*)


r[t_]:={Sin[t],Cos[t],(t^2-t)/10};
angularVel[t_]:=curvature[r[t],t]

tStart=0;
tEnd=6;
sol=integrateAngularVelocity[angularVel];


(* ::Text:: *)
(*Since the initial triad on the helix is not the identity triad we have to add the corresponding reference triad. The reference triad is obtained via a smallest rotation mapping*)


rotationAxis=Normalize[Cross[{1,0,0},D[r[t],t]]/.t->tStart];
rotationAngle=ArcCos[{1,0,0} . Normalize[D[r[t],t]/.t->tStart]];
startQuaternion={Cos[rotationAngle/2],Sin[rotationAngle/2]rotationAxis};


(* ::Text:: *)
(*Integrate the curvature to obtain the original curve*)


tEval=1/2;
rIntegrated=NIntegrate[
  (
    Norm[r'[t]]*R[
      add[startQuaternion,{q0[t],{q1[t], q2[t],q3[t]}}]
    ] . {1,0,0}/.sol
  ),
  {t,tStart,tEval}][[1]]+r[tStart];
Print["position integrated with curvature:\n"<>ToString[rIntegrated]]
Print["position evaluated with parametric curve:\n"<>ToString[N[r[tEval]]]]
