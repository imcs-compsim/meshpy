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
(*Smallest rotation mapping vs integration of the ODE*)


(* ::Text:: *)
(*Differential equation for quaternions (for spatial angular velocity)*)


angularVelocityRHS[q_,\[Omega]_]:=add[q,{0,\[Omega]}]/2


(* ::Text:: *)
(*Integrate angular velocity*)


angularVel[t_]:={2,3+2t,5-2t};
tEnd=1;
sol=NDSolve[
	Join[
		Thread[{q0'[t],q1'[t],q2'[t],q3'[t]}==Flatten@angularVelocityRHS[{q0[t],{q1[t],q2[t],q3[t]}},angularVel[t]]],
		{q0[0]==1,q1[0]==0,q2[0]==0,q3[0]==0}
	],
	{q0,q1,q2,q3},
	{t,0,tEnd}, AccuracyGoal->20,PrecisionGoal->20,WorkingPrecision->35
];
Plot[Evaluate[{q0[t],q1[t],q2[t],q3[t]}/.sol],{t,0,tEnd},PlotStyle->Automatic]
Plot[Evaluate[Norm[{q0[t],q1[t],q2[t],q3[t]}]/.sol],{t,0,tEnd},PlotStyle->Automatic]
{q0[t], q1[t], q2[t],q3[t]}/.sol/.t->tEnd
