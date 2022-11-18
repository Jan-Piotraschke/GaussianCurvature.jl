using Makie  # great package
using GLMakie  # great package
using MeshIO  # good package
using FileIO  # good package
using Meshes
using GeometryBasics  # great package
using Random

# GLMakie.activate!()


# mesh_loaded = load("assets/sphere.stl")

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # PLOTTING DONE BY MAKIE.jl
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# # 3D Code
# scene = Makie.Scene(resolution = (400,400));
# f, ax, pl = Makie.mesh(mesh_loaded, axis=(type=Axis3,))  # plot the mesh
# wireframe!(ax, mesh_loaded, color=(:black, 0.2), linewidth=2, transparency=true)  # only for the asthetic







"""
    calculate_order_parameter(v_order, tt, v_tp)

"""
function calculate_order_parameter(v_order, tt, v_tp)
    for i=1:n
        v_norm[i,:]=v_tp[i,:]/norm(v_tp[i,:])
    end
    v_order[tt]=(1/n)*norm(sum(v_norm))

    return v_order
end


# particle motion on a sphere
n=60  # Number of particles
noise=0.6  # value of the noise


dt=0.01

s=1   # speed

mov=[];

time=400
timesteps = time/dt+1  # %number of timesteps to be calculated
plotstep=0.1  #         %time when data is plotted.
tau=1        #         % relaxation time of orientation

DT=dt;      #        %number of timesteps before nearest neighbour list is calculated

F_rep=1;       #     %maximum repulsive force between particles
R_eq=5/6;     #      %distance for maximum repulsive force betwen particles
R_o=1;      #        %distance for maximum attractive force betwen particles
F_adh=0;     #    %maximum attractive force betwen particles
mu=1;       #        % mobility of the particle

rho=sqrt(n*((R_eq/2)^2)/4); #%radius of sphere acording to "active swarms on a sphere"

Rc=2.4*R_eq/2;            #       %cutoff radius for the calculation of the distance matrix

scale=0.2*rho; # %vector arrow length
u_o=0; # % force perpendicular to the current velocity

# %Intervall for the magnitude of the angle
hi=noise/(2*sqrt(dt));


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# NOTE: angles need to be multiple integers of pi, otherwise the particles fly off due to rounding issues!!!
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

n_list=[0:1:n;]  # generates integer-vector from 0 to n

n_perm = randperm(n)
push!(n_perm, 0)
thet0=(360/n).*n_perm;

n_perm = randperm(n)
push!(n_perm, 0)
phi0=(360/n).*n_perm;
g0=(360/n).*n_perm;

markersize=8;
r=zeros(n,3);
r_norm=zeros(n,3);  # %normalized position vector
r_norm_ini=zeros(n,3);
x=zeros(n,3);
y=zeros(n,3);
z=zeros(n,3);

test=zeros(n,3);

logitude=[];
latitude=[];

DIST=[];
rij=[];
x_pro=0;
y_pro=0;

sphere_radius1=[];
sphere_radius2=[];
x_sphere1=[];
y_sphere1=[];
z_sphere1=[];

Frames2=[]; # %for movie

u_i=0;
phi=[];
thet=[];
phi1=0;
thet1=0;
u = zeros(3,n);  # %allocate force matrix x-direction

a_paper=zeros(n,3); #  % in reference to formula p.3 of phys. rev. E91,022306(2015)
rdot=zeros(n,3);
xdot=zeros(n,3);
ydot=zeros(n,3);
zdot=zeros(n,3);
v=zeros(n,3);

v_order=zeros(n,1);  #%order parameter
v_norm= zeros(n,3); # %normalized velocity


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# initial conditions:
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

r0=zeros(n,3)
x01=zeros(n,1)
x02=zeros(n,1)
x03=zeros(n,1)
y01=zeros(n,1)
y02=zeros(n,1)
y03=zeros(n,1)
z01=zeros(n,1)
z02=zeros(n,1)
z03=zeros(n,1)

for i=1:n

    r0[i,:]=rho*[sind(phi0[i])*cosd(thet0[i]); sind(phi0[i])*sind(thet0[i]); cosd(phi0[i])];  # % starting position in cartesian

    # %initial orientation:
    x01[i,1]=cosd(g0[i])*cosd(phi0[i])*cosd(thet0[i])-sind(g0[i])*sind(thet0[i]);
    x02[i,1]=cosd(g0[i])*cosd(phi0[i])*sind(thet0[i])+sind(g0[i])*cosd(thet0[i]);
    x03[i,1]=-cosd(g0[i])*sind(thet0[i]);

    y01[i,1]=-sind(g0[i])*cosd(phi0[i])*cosd(thet0[i])-cosd(g0[i])*sind(thet0[i]);
    y02[i,1]=-sind(g0[i])*cosd(phi0[i])*sind(thet0[i])+cosd(g0[i])*cosd(thet0[i]);
    y03[i,1]=sind(g0[i])*sind(phi0[i]);

    z01[i,1]=sind(phi0[i])*cosd(thet0[i]);
    z02[i,1]=sind(phi0[i])*sind(thet0[i]);
    z03[i,1]=cosd(phi0[i]);

end

r=r0;
x=zeros(n,3)
y=zeros(n,3)
z=zeros(n,3)

x[:,1]=x01;
x[:,2]=x02;
x[:,3]=x03;
y[:,1]=y01;
y[:,2]=y02;
y[:,3]=y03;
z[:,1]=z01;
z[:,2]=z02;
z[:,3]=z03;


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# graphic output
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# %NOTE: making movies containing surface plots in MatLab will only work if the monitor is set to 16bit instead of 32bit!!

fps = 25;
writerObj = VideoWriter('290415_Forcetest4_Sphere_OP_N60_rho_5_Frep_1_Fadh_0_s1_t400_fps25.mp4','MPEG-4'); % Open the video writer object
writerObj.FrameRate = fps;
writerObj.Quality = 100;
open(writerObj);




for tt=2:timesteps #   %number of time steps
    Distmat=zeros(n,n)  # %allocate memory for distance matrix
    Distmat2=zeros(n,n)  # %allocate memory for second distance matrix

    t=(tt-1)*dt;
    

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # plot the position of each particle on the specific time point
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    if rem(t,plotstep)==0
        
        hFig = figure(1);

        # # resize the image window
        # set(hFig, 'Position', [200 100 1000 800])
        
        
        subplot(2,2,1) . #  %plot the sphere and particles in 3D

        [xs,ys,zs]=sphere(100);
        ss=1;

        h1=surf(ss*rho*xs,ss*rho*ys,ss*rho*zs);

        set (h1,'EdgeColor',[0.75,0.75,0.75],'FaceColor',[0.95,0.95,0.75],'MeshStyle','row');

        alpha(0.5);

        
        # %particle position:
        plot3(r(:,1),r(:,2),r(:,3),'o','MarkerSize', 8,'MarkerEdgeColor','k','MarkerFaceColor','g');
        plot3(r(1,1),r(1,2),r(1,3),'o','MarkerSize', 8,'MarkerEdgeColor','k','MarkerFaceColor','r');
        [xs2,ys2,zs2]=sphere(20);
        h2=surf(ss*R_eq/2*xs2+r(1,1),ss*R_eq/2*ys2+r(1,2),ss*R_eq/2*zs2+r(1,3));
        set (h2,'EdgeColor',[1,0,0],'FaceColor',[1,0,0],'MeshStyle','row');
        alpha(0.2);
        
        
        axis([-rho  rho  -rho  rho  -rho  rho]);
        title(['tt:' num2str(tt,'%.3e'),' of ' num2str(timesteps,'%.e'), '; dt:' num2str(dt,'%.e'),'; F_{rep}:' num2str(F_rep),'; F_{adh}:' num2str(F_adh),'; Rho:' num2str(rho),'; s:' num2str(s)]);
        
        # %tangent vector (T); red:
        quiver3(r(:,1),r(:,2),r(:,3),scale*a_paper(:,1),scale*a_paper(:,2),scale*a_paper(:,3),0,'MaxHeadSize', .8,'color','r');
        
        # %vector perpendicular to tangent and normal vector (TxN); black:
        quiver3(r(:,1),r(:,2),r(:,3),v_norm(:,1),v_norm(:,2),v_norm(:,3),0,'MaxHeadSize', .8,'color','m');
        
        # %orientation of the particle (O); black:
        quiver3(r(:,1),r(:,2),r(:,3),x(:,1),x(:,2),x(:,3),0,'MaxHeadSize', .8,'color','k');

        # %normal vector (N); blue:
        quiver3(r(:,1),r(:,2),r(:,3),scale*r_norm_ini(:,1),scale*r_norm_ini(:,2),scale*r_norm_ini(:,3),0,'MaxHeadSize', .8,'color','b');
        
        
        subplot(2,2,[3 4]); # %EQUIRECTANGULAR PROJECTION

        plot(thet1(1,:),phi1(1,:),'o','MarkerSize', 8,'MarkerEdgeColor','k','MarkerFaceColor','g');
        
        plot(thet1(1,1),phi1(1,1),'o','MarkerSize', 8,'MarkerEdgeColor','k','MarkerFaceColor','r');
        
        axis([-pi  pi  -pi/2  pi/2]);
        
        title(['N:' num2str(n), '  Theta:' num2str(thet1(1,1),'%.2f'), '  Phi:' num2str(phi1(1,1),'%.2f'), ' |R|:' num2str(norm(r(1,:))),'  x:' num2str(r(1,1),'%.1f'),'  y:'  num2str(r(1,2),'%.1f'),'  z:' num2str(r(1,3),'%.1f')]);
        xlabel('Theta (azimuth) -pi < x < pi')  # % x-axis label
        ylabel('Phi (altitude) -pi/2 < y < pi/2') # % y-axis label

        subplot(2,2,2)  # %ORDER PARAMETER

        plot(v_order)
        title(['mean order parameter:', num2str(mean(v_order),'%.3f')]);
        xlabel('timestep')
        ylabel('order parameter')

        frame = getframe(hFig);
        writeVideo(writerObj,frame);
        
        %     end
    end
    

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    # %----------------------------------------------------------------------
    # %Generate Distance Matrix DistMat:
    # %----------------------------------------------------------------------
    
    if rem(t,DT)==0
        for i=1:n
            for j=i+1:n
                # %arc-length between particles:
                p_dist1=r(j,:)-r(i,:);
                DIST=norm(p_dist1);
                Distmat(i,j)=DIST;
                
                # %compute the elements within cutoff range Rc
                if (DIST<Rc)&&(i~=j)
                    Distmat2(i,j)=1;    # %Matrix specifying particles within range Rc
                end
            end
        end
    end
    
    # %--------------------------------------------------------------------------
    # %%%%%%%%%%%%%%%%%%%%% Force Calcualtions %%%%%%%%%%%%%%%%%%%%%%%%%%
    # %--------------------------------------------------------------------------
    
    for i_i=1:n
        
        # %calculating forces for particle i
        # %---------------------------------------------------------------
        a=find(Distmat2(i_i,:));  #%finds closest particles
        [bb numnearestneighbs]=size(a); # %extract entries of the row-vector size(a)
        sum_cross_prod = [0,0,0];
        
        if isempty(a)   # %when there are no particles within range Rc...
            # % ...no forces from other particles.
            u(1) = 0;
            u(2) = 0;
            u(3) = 0;
            
        else

            u=zeros(3,numnearestneighbs);

            if tt == 100000
                Distmat2(i_i,a)
                figure(200)
                plot3(r(a(:),1),r(a(:),2),r(a(:),3),'b*')
                ss=1;
                [xs,ys,zs]=sphere(100);
                h2=surf(ss*rho*xs,ss*rho*ys,ss*rho*zs);

                set (h2,'EdgeColor',[0.75,0.75,0.75],'FaceColor',[0.95,0.95,0.75],'MeshStyle','row');
                alpha(0.5);
                
                plot3(r(i_i,1),r(i_i,2),r(i_i,3),'r*')
                [xs2,ys2,zs2]=sphere(100);
                h2=surf(ss*Rc*xs2+r(i_i,1),ss*Rc*ys2+r(i_i,2),ss*Rc*zs2+r(i_i,3));

                set (h2,'EdgeColor',[0.75,0.75,0.75],'FaceColor',[0.95,0.95,0.75],'MeshStyle','row');

                alpha(0.5);

                [xs2,ys2,zs2]=sphere(100);
                h2=surf(ss*R_eq/2*xs2+r(i_i,1),ss*R_eq/2*ys2+r(i_i,2),ss*R_eq/2*zs2+r(i_i,3));

                set (h2,'EdgeColor',[1,0,0],'FaceColor',[1,0,0],'MeshStyle','row');

                alpha(0.5);

            end
            for k=1:numnearestneighbs
                p_dist=r(a(k),:)-r(i_i,:); # % vector between particle pairs

                sum_cross_prod = sum_cross_prod+cross(x(a(k),:),x(i_i,:));
                # %distance between particles
                rij=norm(p_dist);
                # % calculate the force from this distance here!
                # %------------------------------------------------------------------

                # %compute unit normal vector, rdiffnorm(1)=n_x, rdiffnorm(2)=n_y
                rdiffnorm = p_dist./rij;
                # %----------------------------------------------------------
                # %              Setting the Force conditions
                # %----------------------------------------------------------
                if rij>R_eq
                    # %no force acting on particle if rij>R_o
                    u(:,k) = [0; 0; 0];
                elseif rij>R_eq  # %attractive force  (if rij>R_eq)
                    u(:,k) = rdiffnorm.*F_adh.*((rij-R_eq)./(R_o-R_eq));
                else #  %repulsive force (if rij<R_eq)
                    # %repulsive force from viscek model:
                    # %u(:,k) = rdiffnorm.*F_rep.*((rij-R_eq)./R_eq);
                    # %repulsive force from the paper "active swarms on a sphere":
                    u(:,k) = rdiffnorm.*F_rep.*(rij-R_eq);
                end
            end
        end

        # %------------------------------------------------------------------
        # %Compute the position and velocity
        # %------------------------------------------------------------------

        # %particle velocities:
        # %--------------------

        # %compute the intermediate vector a (velocity without correction of 3D)
        a_paper(i_i,:)=s.*x(i_i,:) + mu.*[sum(u(1,:)) sum(u(2,:)) sum(u(3,:))];
        # %compute the new velocity of the particle rdot
        
        r_norm(i_i,:)=r(i_i,:)./norm(r(i_i,:));  #%normalized position vector
        
        # %Normal vector look strange, they are rotating somehow. Will try and
        # %use the normal vector calculate above to be displayed in the video
        r_norm_ini(i_i,:)=r_norm(i_i,:);
        rdot(i_i,:)=a_paper(i_i,:)-dot(r_norm(i_i,:),a_paper(i_i,:))*r_norm(i_i,:);
        

        # %change of the direction of the self-propelling velocity divided by
        # %time ndot (here xdot) in the paper (in reference to formula p.3 of phys. rev. E91,022306(2015))
        # %        ----
        xdot(i_i,:)=-tau*dot(r_norm(i_i,:),sum_cross_prod)*cross(r_norm(i_i,:),x(i_i,:));

        # %----------------------------
        # %calculate the new positions:
        # %----------------------------

        # %verlet integration:
        r(i_i,:)=r(i_i,:)+rdot(i_i,:)*dt;% + 0.5.*xdot(i_i,:).*dt.*dt;# %position in cartisian coordinates

        r_norm(i_i,:)=r(i_i,:)./norm(r(i_i,:));# %normalized position vector
        
        x(i_i,:)=x(i_i,:)+xdot(i_i,:)*dt;

        # backprojection of the current position and x onto the new plane
        r(i_i,:)=r(i_i,:)/norm(r(i_i,:))*rho;
        
        x(i_i,:)=(x(i_i,:)-dot(r_norm(i_i,:),x(i_i,:))*r_norm(i_i,:))./norm(x(i_i,:)-dot(r_norm(i_i,:),x(i_i,:))*r_norm(i_i,:));
    
        # %------------------------------------------------------------
        # %multiple particles coord.:
        # %------------------------------------------------------------
        # %         needed for Map-projection...
        r_p(i_i) = sqrt(r(i_i,1)^2 + r(i_i,2)^2 + r(i_i,3)^2);
        phi(i_i) = asin(r(i_i,3)/r_p(i_i));   #% elevation angle
        thet(i_i) = atan2(r(i_i,2),r(i_i,1)); #% azimuth
        
        thet1=thet;
        phi1=phi;
        
        # %--------------------------------------------------------------------------
        # %                 Order Parameter Calculation
        # %--------------------------------------------------------------------------


        # %order Parameter

        # %vector perpendicular to tangent and position vector
        v_tp(i_i,:)=cross(r(i_i,:),rdot(i_i,:));

    end
    
    # calculate the order parameter    
    v_order = calculate_order_parameter(v_order, tt, v_tp)

end
close(writerObj);


