import numpy as np

class multi_cartpole():
    def __init__(self,N_carts):
        self.N_carts = N_carts
        self.viewer = None

    def render(self, state, mode='human'):
        screen_width = 600
        screen_height = 400

        world_width = 10 * 2
        scale = screen_width/world_width
        carty = 100  # TOP OF CART
        polewidth = 10.0
        polelen = scale * 2
        cartwidth = 50.0
        cartheight = 30.0

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)

            self.carttrans = list()
            self.poletrans = list()
            for i in range(self.N_carts):
                l, r, t, b = -cartwidth / 2, cartwidth / 2, cartheight / 2, -cartheight / 2
                axleoffset = cartheight / 4.0
                cart = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
                self.carttrans.append(rendering.Transform())
                cart.add_attr(self.carttrans[i])
                self.viewer.add_geom(cart)
                l, r, t, b = -polewidth / 2, polewidth / 2, polelen - polewidth / 2, -polewidth / 2
                pole = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
                pole.set_color(.8, .6, .4)
                self.poletrans.append(rendering.Transform(translation=(0, axleoffset)))
                pole.add_attr(self.poletrans[i])
                pole.add_attr(self.carttrans[i])
                self.viewer.add_geom(pole)
                self.axle = rendering.make_circle(polewidth/2)
                self.axle.add_attr(self.poletrans[i])
                self.axle.add_attr(self.carttrans[i])
                self.axle.set_color(.5, .5, .8)
                self.viewer.add_geom(self.axle)
                self.track = rendering.Line((0, carty), (screen_width, carty))
                self.track.set_color(0, 0, 0)
                self.viewer.add_geom(self.track)

            # self._pole_geom = pole

        # Edit the pole polygon vertex
        # pole = self._pole_geom
        # l, r, t, b = -polewidth / 2, polewidth / 2, polelen - polewidth / 2, -polewidth / 2
        # pole.v = [(l, b), (l, t), (r, t), (r, b)]
        i = 0
        for obj1, obj2 in zip(self.carttrans, self.poletrans):
            cartx = state[i,0] * scale + screen_width / 2.0  # MIDDLE OF CART
            obj1.set_translation(cartx, carty)
            obj2.set_rotation(-state[i,2])
            i += 1

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')
